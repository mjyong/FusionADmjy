#!/usr/bin/env python
"""
FusionAD Stage1 (Track+Map) 训练流程 Debug & 耗时统计

调用链路:
  tools/train.py main()
    -> Config.fromfile() -> importlib(plugin) -> build_model(FusionAD)
    -> custom_train_model() -> custom_train_detector()
       -> build_dataloader, build_optimizer, build_runner
       -> runner.run():
            每个iter:
              DataLoader -> NuScenesE2EDataset.prepare_train_data()
                -> 5帧连续帧 -> pipeline -> union2one()
              FusionAD.forward_train():
                A. forward_track_train(遍历5帧):
                   _forward_single():
                     get_bevs():
                       get_history_bev(prev) [no_grad]
                       extract_feat(img) [freeze_img_modules -> no_grad]
                         GridMask -> img_backbone(R101) -> img_neck(FPN)
                       extract_pts_feat(points)
                         voxelize [no_grad] -> pts_backbone(SparseEncoderHD)
                       pts_bbox_head.get_bev_features()
                         BEVFormerEncoder x6 (PtsCrossAttn+SpatialCrossAttn+TemporalSelfAttn)
                     pts_bbox_head.get_detections()
                       DetectionTransformerDecoder x6
                     criterion.match_for_single_frame() (匈牙利匹配)
                     memory_bank() -> query_interact() (QIM)
                B. seg_head.forward_train(bev_embed)
                   PansegformerHead -> SegDeformableTransformer
              backward() + clip_grad_norm_(35) + optimizer.step()

Stage1配置: FusionAD(Track+Map), Camera+LiDAR, freeze_img_modules=True,
            AdamW lr=1.2e-4, queue_length=5, 20 epochs
"""

from __future__ import division
import argparse
import copy
import os
import sys
import time
import warnings
from collections import defaultdict

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")

# 抑制mmcv/mmdet的冗余日志
import logging
for _name in ['mmcv', 'mmdet', 'mmdet3d', 'mmseg', 'root']:
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description='FusionAD Stage1 Debug')
    parser.add_argument(
        'config', nargs='?',
        default='projects/configs/stage1_track_map/fusion_base_track_map.py')
    parser.add_argument('--work-dir', default='./work_dirs/debug_stage1')
    parser.add_argument('--debug-iters', type=int, default=3)
    parser.add_argument('--no-data', action='store_true',
                        help='使用mock数据, 不需要真实数据集')
    parser.add_argument('--profile', action='store_true',
                        help='开启各阶段耗时统计')
    parser.add_argument('--check-grad', action='store_true',
                        help='检查梯度流')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


# =============================================================================
# 耗时统计器
# =============================================================================

class StageTimer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.timings = defaultdict(list)
        self._starts = {}

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name):
        if not self.enabled:
            return
        self._sync()
        self._starts[name] = time.time()

    def end(self, name):
        if not self.enabled or name not in self._starts:
            return
        self._sync()
        self.timings[name].append((time.time() - self._starts[name]) * 1000)

    def report(self, title="耗时统计"):
        if not self.enabled or not self.timings:
            return
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")
        order = [
            'data_loading', 'forward_total',
            '  forward_track_train',
            '    extract_img_feat', '    extract_pts_feat',
            '    get_bev_features', '    get_detections',
            '  seg_head.forward_train',
            'backward', 'optimizer_step', 'iter_total',
        ]
        all_keys = list(self.timings.keys())
        for k in all_keys:
            if k not in order:
                order.append(k)

        iter_avg = 0
        rows = []
        for name in order:
            if name not in self.timings:
                continue
            vals = self.timings[name]
            avg = sum(vals) / len(vals)
            rows.append((name, avg, min(vals), max(vals), len(vals)))
            if name == 'iter_total':
                iter_avg = avg

        for name, avg, mn, mx, cnt in rows:
            pct = avg / max(iter_avg, 1e-9) * 100 if iter_avg > 0 else 0
            print(f"  {name:35s}  avg={avg:8.1f}ms  "
                  f"min={mn:7.1f}  max={mx:7.1f}  "
                  f"({pct:5.1f}%)  [{cnt}x]")
        print(f"{'='*70}")


# =============================================================================
# 模型分析
# =============================================================================

def analyze_model_structure(model):
    print(f"\n{'='*80}")
    print(f" 模型结构分析")
    print(f"{'='*80}")
    total_p, train_p = 0, 0
    for name, module in model.named_children():
        mt = sum(p.numel() for p in module.parameters())
        mr = sum(p.numel() for p in module.parameters() if p.requires_grad)
        total_p += mt
        train_p += mr
        mode = 'TRAIN' if module.training else 'EVAL'
        print(f"  {name:30s}  total={mt:>12,d}  trainable={mr:>12,d}  "
              f"({mr/max(mt,1)*100:5.1f}%)  [{mode}]")
    print(f"\n  {'TOTAL':30s}  total={total_p:>12,d}  trainable={train_p:>12,d}  "
          f"({train_p/max(total_p,1)*100:.1f}%)")
    print(f"{'='*80}")


def check_freeze_status(model):
    print("\n[冻结状态]")
    def _check(name, mod, expect_frozen):
        if mod is None:
            return
        n_train = sum(1 for p in mod.parameters() if p.requires_grad)
        n_total = sum(1 for _ in mod.parameters())
        is_eval = not mod.training
        ok = (n_train == 0 or is_eval) if expect_frozen else (n_train > 0)
        tag = "OK" if ok else "WARN"
        print(f"  [{tag}] {name}: {n_train}/{n_total} trainable, "
              f"{'eval' if is_eval else 'train'}")

    _check('img_backbone', getattr(model, 'img_backbone', None), True)
    _check('img_neck', getattr(model, 'img_neck', None), True)
    _check('pts_backbone', getattr(model, 'pts_backbone', None), False)
    _check('pts_bbox_head', getattr(model, 'pts_bbox_head', None), False)
    _check('seg_head', getattr(model, 'seg_head', None), False)


# =============================================================================
# 梯度/Loss检查
# =============================================================================

def check_gradient_flow(model):
    print("\n[梯度流]")
    stats = defaultdict(lambda: {'ok': 0, 'none': 0, 'nan': 0, 'max': 0.0})
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mod = name.split('.')[0]
        s = stats[mod]
        if p.grad is not None:
            gn = p.grad.data.norm(2).item()
            s['ok'] += 1
            s['max'] = max(s['max'], gn)
            if torch.isnan(p.grad).any():
                s['nan'] += 1
        else:
            s['none'] += 1
    for mod in sorted(stats):
        s = stats[mod]
        tag = "NaN!" if s['nan'] else ("无梯度!" if s['ok'] == 0 else "OK")
        print(f"  {mod:25s}  有梯度={s['ok']:>4d}  无={s['none']:>4d}  "
              f"max_norm={s['max']:.4e}  [{tag}]")


def check_loss_values(losses, iter_idx):
    print(f"\n  [Loss] iter={iter_idx}")
    task_sums = defaultdict(float)
    for k in sorted(losses.keys()):
        v = losses[k]
        val = v.item() if isinstance(v, torch.Tensor) else v
        task_sums[k.split('.')[0]] += val
        flag = ""
        if val != val:
            flag = " NaN!"
        elif abs(val) > 1e5:
            flag = " 极大!"
        if flag:
            print(f"    {k:40s} = {val:>10.4f}{flag}")

    total = 0
    for task in sorted(task_sums):
        print(f"    {task:12s}: {task_sums[task]:>10.4f}")
        total += task_sums[task]
    print(f"    {'TOTAL':12s}: {total:>10.4f}")
    return total


# =============================================================================
# Monkey-patch计时
# =============================================================================

def patch_timing(model, timer):
    actual = model.module if hasattr(model, 'module') else model

    def _wrap(obj, attr, timer_name):
        orig = getattr(obj, attr)
        def wrapper(*a, **kw):
            timer.start(timer_name)
            r = orig(*a, **kw)
            timer.end(timer_name)
            return r
        setattr(obj, attr, wrapper)

    _wrap(actual, 'extract_img_feat', '    extract_img_feat')
    _wrap(actual, 'extract_pts_feat', '    extract_pts_feat')
    _wrap(actual.pts_bbox_head, 'get_bev_features', '    get_bev_features')
    _wrap(actual.pts_bbox_head, 'get_detections', '    get_detections')
    _wrap(actual, 'forward_track_train', '  forward_track_train')
    if actual.with_seg_head:
        _wrap(actual.seg_head, 'forward_train', '  seg_head.forward_train')

    orig_ft = actual.forward_train
    def timed_ft(*a, **kw):
        timer.start('forward_total')
        r = orig_ft(*a, **kw)
        timer.end('forward_total')
        return r
    actual.forward_train = timed_ft


# =============================================================================
# Config覆盖 (mock模式, 减小模型以适应24GB显存)
# =============================================================================

def override_cfg_for_debug(cfg):
    """
    覆盖config中的关键参数, 使模型可在24GB GPU上用mock数据运行。
    主要缩小: BEV分辨率 200x200->50x50, encoder层数 6->2, queue_length 5->1
    """
    # BEV分辨率: 200x200 -> 50x50
    small_bev_h, small_bev_w = 50, 50

    cfg.queue_length = 1  # 只用1帧, 大幅省显存

    # pts_bbox_head
    head = cfg.model.pts_bbox_head
    head.bev_h = small_bev_h
    head.bev_w = small_bev_w

    # encoder: 6->2 layers
    head.transformer.encoder.num_layers = 2
    # decoder: 6->2 layers
    head.transformer.decoder.num_layers = 2

    # positional encoding
    head.positional_encoding.row_num_embed = small_bev_h
    head.positional_encoding.col_num_embed = small_bev_w

    # seg_head
    if hasattr(cfg.model, 'seg_head') and cfg.model.seg_head is not None:
        seg = cfg.model.seg_head
        seg.bev_h = small_bev_h
        seg.bev_w = small_bev_w
        seg.canvas_size = (small_bev_h, small_bev_w)
        # seg encoder/decoder layers
        if hasattr(seg, 'transformer'):
            seg.transformer.encoder.num_layers = 2
            seg.transformer.decoder.num_layers = 2

    # 其它可能引用bev_size的head
    for head_name in ['motion_head', 'occ_head', 'planning_head']:
        h = getattr(cfg.model, head_name, None)
        if h is not None and isinstance(h, dict):
            if 'bev_size' in h:
                h['bev_size'] = (small_bev_h, small_bev_w)

    print(f"  [Override] bev={small_bev_h}x{small_bev_w}, "
          f"encoder=2L, decoder=2L, queue=1")
    return cfg


# =============================================================================
# Mock数据生成器
# =============================================================================

def mock_data_iterator(cfg, device, n_iters=5):
    """
    生成与 NuScenesE2EDataset.union2one() collate后格式一致的mock数据。
    配合 override_cfg_for_debug() 使用小BEV/小图像以避免24GB显存OOM。
    """
    import numpy as np
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    queue_length = cfg.get('queue_length', 1)  # 已被override为1
    num_cams = 6
    H, W = 128, 192  # 极小分辨率
    num_gt = 3
    past_steps = cfg.get('past_steps', 4)
    fut_steps = cfg.get('fut_steps', 4)
    traj_steps = past_steps + fut_steps
    bev_h = cfg.model.pts_bbox_head.get('bev_h', 50)
    bev_w = cfg.model.pts_bbox_head.get('bev_w', 50)
    num_pts = 5000

    for _ in range(n_iters):
        metas = {}
        for fi in range(queue_length):
            metas[fi] = {
                'scene_token': 'mock_scene',
                'prev_bev': fi > 0,
                'can_bus': np.zeros(18, dtype=np.float64),
                'lidar2img': [np.eye(4, dtype=np.float32) for _ in range(num_cams)],
                'img_shape': [(H, W, 3)] * num_cams,
                'box_type_3d': LiDARInstance3DBoxes,
            }

        data = {}
        data['img'] = torch.randn(1, queue_length, num_cams, 3, H, W, device=device)
        data['img_metas'] = [metas]
        data['points'] = [[
            torch.randn(num_pts, 5, device=device) for _ in range(queue_length)
        ]]

        gt_boxes, gt_labels, gt_inds = [], [], []
        gt_past_traj, gt_past_traj_mask = [], []
        gt_sdc_bbox, gt_sdc_label = [], []

        for fi in range(queue_length):
            # box_dim=9: x,y,z,dx,dy,dz,yaw,vx,vy (与真实数据一致)
            boxes_np = torch.randn(num_gt, 9).numpy()
            gt_boxes.append(LiDARInstance3DBoxes(boxes_np, box_dim=9))
            gt_labels.append(torch.randint(0, 10, (num_gt,), device=device))
            gt_inds.append(torch.arange(1, num_gt + 1, device=device))
            # 模型past_traj_reg_branches输出shape=[N, past_steps+fut_steps, 2]
            gt_past_traj.append(torch.randn(num_gt, traj_steps, 2, device=device))
            gt_past_traj_mask.append(torch.ones(num_gt, traj_steps, 2, device=device))
            sdc_np = torch.randn(1, 9).numpy()
            gt_sdc_bbox.append(LiDARInstance3DBoxes(sdc_np, box_dim=9))
            gt_sdc_label.append(torch.zeros(1, dtype=torch.long, device=device))

        data['gt_bboxes_3d'] = [gt_boxes]
        data['gt_labels_3d'] = [gt_labels]
        data['gt_inds'] = [gt_inds]
        data['gt_past_traj'] = [gt_past_traj]
        data['gt_past_traj_mask'] = [gt_past_traj_mask]
        data['gt_sdc_bbox'] = [gt_sdc_bbox]
        data['gt_sdc_label'] = [gt_sdc_label]
        data['l2g_r_mat'] = [[torch.eye(3, device=device) for _ in range(queue_length)]]
        data['l2g_t'] = [[torch.zeros(3, device=device) for _ in range(queue_length)]]
        data['timestamp'] = [[torch.tensor(fi * 0.5, device=device)
                               for fi in range(queue_length)]]
        data['gt_fut_traj'] = [torch.randn(num_gt, 12, 2, device=device)]
        data['gt_fut_traj_mask'] = [torch.ones(num_gt, 12, device=device)]
        data['gt_sdc_fut_traj'] = [torch.randn(1, 12, 2, device=device)]
        data['gt_sdc_fut_traj_mask'] = [torch.ones(1, 12, device=device)]
        data['gt_lane_labels'] = [torch.randint(0, 4, (20,), device=device)]
        data['gt_lane_bboxes'] = [torch.randn(20, 4, device=device)]
        data['gt_lane_masks'] = [torch.randint(0, 2, (20, bev_h, bev_w),
                                               device=device).float()]
        # Stage1不需要的字段
        data['gt_segmentation'] = None
        data['gt_instance'] = None
        data['gt_occ_img_is_valid'] = None
        data['sdc_planning'] = None
        data['sdc_planning_mask'] = None
        data['command'] = None
        data['gt_future_boxes'] = None

        yield data


# =============================================================================
# 主入口
# =============================================================================

def main():
    args = parse_args()

    # --- Step 1: Config ---
    try:
        from mmcv import Config
    except ImportError:
        print("[ERROR] mmcv未安装")
        return

    cfg = Config.fromfile(args.config)
    print(f"[Config] {args.config}")
    print(f"  model={cfg.model.type}, queue={cfg.get('queue_length','?')}, "
          f"epochs={cfg.get('total_epochs','?')}, "
          f"lr={cfg.optimizer.lr}, bs={cfg.data.samples_per_gpu}")

    # --- Step 2: Plugin ---
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        _module_path = os.path.dirname(cfg.plugin_dir).replace('/', '.')
        plg_lib = importlib.import_module(_module_path)

    # --- Step 2.5: Mock模式缩小模型 ---
    if args.no_data:
        cfg = override_cfg_for_debug(cfg)

    # --- Step 3: 构建模型 ---
    from mmdet3d.models import build_model
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    model = model.to(device)
    model.train()
    print(f"[Model] 构建成功, device={device}")

    analyze_model_structure(model)
    check_freeze_status(model)

    # --- Step 4: 优化器 ---
    from mmcv.runner import build_optimizer
    optimizer = build_optimizer(model, cfg.optimizer)
    print(f"\n[Optimizer] {cfg.optimizer.type}, "
          f"{len(optimizer.param_groups)} param groups")

    # --- Step 5: 数据 ---
    timer = StageTimer(enabled=args.profile)

    if args.no_data:
        print("[Data] Mock模式")
        data_iter = mock_data_iterator(cfg, device, n_iters=args.debug_iters)
    else:
        try:
            from mmdet3d.datasets import build_dataset
            from projects.mmdet3d_plugin.datasets.builder import build_dataloader
            dataset = build_dataset(cfg.data.train)
            print(f"[Data] 数据集大小: {len(dataset)}")
            dataloader = build_dataloader(
                dataset,
                cfg.data.samples_per_gpu,
                workers_per_gpu=0,  # 单线程, 方便调试
                num_gpus=1,
                dist=False,
                seed=args.seed,
                shuffler_sampler=cfg.data.get(
                    'shuffler_sampler', dict(type='DistributedGroupSampler')),
                nonshuffler_sampler=cfg.data.get(
                    'nonshuffler_sampler', dict(type='DistributedSampler')),
            )
            data_iter = iter(dataloader)
        except Exception as e:
            print(f"[Data] 加载失败({e}), 切换Mock模式")
            data_iter = mock_data_iterator(cfg, device, n_iters=args.debug_iters)

    # --- Step 5.5: 注入计时 ---
    if args.profile:
        patch_timing(model, timer)

    # --- Step 6: 训练循环 (FP32, deformable attn不支持FP16) ---
    print(f"\n[Train] 开始 {args.debug_iters} iters")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for it in range(args.debug_iters):
        # 数据加载
        timer.start('data_loading')
        try:
            data = next(data_iter)
        except StopIteration:
            print(f"  iter {it}: 数据耗尽")
            break
        timer.end('data_loading')

        timer.start('iter_total')

        # 前向
        try:
            losses = model.forward(return_loss=True, **data)
        except Exception as e:
            print(f"  iter {it}: 前向失败 - {e}")
            import traceback; traceback.print_exc()
            break

        # Loss
        check_loss_values(losses, it)

        # 反向
        timer.start('backward')
        optimizer.zero_grad()
        total_loss = sum(v for v in losses.values()
                         if isinstance(v, torch.Tensor) and v.requires_grad)
        try:
            total_loss.backward()
        except Exception as e:
            print(f"  iter {it}: 反向失败 - {e}")
            import traceback; traceback.print_exc()
            break
        timer.end('backward')

        # 梯度裁剪
        grad_clip = cfg.optimizer_config.get('grad_clip', None)
        if grad_clip:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), **grad_clip)
            print(f"  grad_norm={gn:.2f} (clip={grad_clip['max_norm']})")

        # 梯度检查
        if args.check_grad and it == 0:
            check_gradient_flow(model)

        # 更新
        timer.start('optimizer_step')
        optimizer.step()
        timer.end('optimizer_step')

        timer.end('iter_total')

        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  iter {it}: loss={total_loss.item():.4f}, peak_mem={mem:.2f}GB")

    # --- Step 7: 汇总 ---
    if args.profile:
        timer.report(f"FusionAD Stage1 耗时 ({args.debug_iters} iters)")

    if torch.cuda.is_available():
        print(f"\nGPU {args.gpu_id} 峰值显存: "
              f"{torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
    print("Done.")


if __name__ == '__main__':
    main()
