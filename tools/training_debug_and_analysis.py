#!/usr/bin/env python
"""
FusionAD Stage1 (Track+Map) 多卡训练Debug & 细粒度耗时统计

启动方式 (8卡):
  # 真实数据 (自动is_debug=True, 只加载少量样本)
  bash tools/dist_train.sh projects/configs/stage1_track_map/fusion_base_track_map.py 8 \
       --cfg-options data.train.is_debug=True data.train.len_debug=50

  # 本脚本 (多卡, 真实数据, 少量iter)
  python -m torch.distributed.launch --nproc_per_node=8 \
      tools/training_debug_and_analysis.py --profile --check-grad --debug-iters 5

  # 本脚本 (多卡, mock数据, 不需要真实数据集)
  python -m torch.distributed.launch --nproc_per_node=8 \
      tools/training_debug_and_analysis.py --no-data --profile --debug-iters 3

  # 单卡调试
  python tools/training_debug_and_analysis.py --no-data --profile --debug-iters 3
"""

from __future__ import division
import argparse
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist

warnings.filterwarnings("ignore")

import logging
for _name in ['mmcv', 'mmdet', 'mmdet3d', 'mmseg', 'root']:
    logging.getLogger(_name).setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(description='FusionAD Stage1 Debug (多卡)')
    parser.add_argument(
        'config', nargs='?',
        default='projects/configs/stage1_track_map/fusion_base_track_map.py')
    parser.add_argument('--work-dir', default='./work_dirs/debug_stage1')
    parser.add_argument('--debug-iters', type=int, default=5)
    parser.add_argument('--debug-samples', type=int, default=80,
                        help='is_debug模式加载的样本数')
    parser.add_argument('--no-data', action='store_true',
                        help='使用mock数据, 不需要真实数据集')
    parser.add_argument('--profile', action='store_true',
                        help='细粒度耗时统计')
    parser.add_argument('--check-grad', action='store_true',
                        help='检查梯度流')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


# =========================================================================
# 分布式工具
# =========================================================================

def setup_dist(launcher):
    if launcher == 'none':
        return False, 0, 1
    from mmcv.runner import init_dist, get_dist_info
    init_dist(launcher, backend='nccl')
    rank, world_size = get_dist_info()
    return True, rank, world_size


def log_rank0(msg, rank=0):
    if rank == 0:
        print(msg)


# =========================================================================
# 细粒度耗时统计器
# =========================================================================

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
        self.timings[name].append((time.time() - self._starts.pop(name)) * 1000)

    def report(self, title="耗时统计"):
        if not self.enabled or not self.timings:
            return
        print(f"\n{'='*78}")
        print(f" {title}")
        print(f"{'='*78}")
        order = [
            'data_loading',
            'forward_total',
            '  forward_track_train',
            '    extract_img_feat',
            '      gridmask',
            '      img_backbone',
            '      img_neck',
            '    extract_pts_feat',
            '      voxelize',
            '      pts_backbone',
            '    get_bev_features',
            '      bev_encoder',
            '    get_detections',
            '      det_decoder',
            '    match_single_frame',
            '    memory_bank',
            '    query_interact',
            '  seg_head.forward_train',
            '    seg_encoder',
            '    seg_decoder',
            '    seg_loss',
            'backward',
            'grad_clip',
            'optimizer_step',
            'iter_total',
        ]
        for k in self.timings:
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
        print(f"{'='*78}")


# =========================================================================
# 模型分析
# =========================================================================

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
    print(f"  显存(参数): {total_p*4/1024**3:.2f}GB total, "
          f"{train_p*4/1024**3:.2f}GB trainable")
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

    for attr in ['freeze_track', 'freeze_seg']:
        val = getattr(model, attr, None)
        if val is not None:
            warn = " ← Stage1应为False!" if val else ""
            print(f"  [INFO] {attr}={val}{warn}")


# =========================================================================
# 梯度/Loss检查
# =========================================================================

def check_gradient_flow(model):
    actual = model.module if hasattr(model, 'module') else model
    print("\n[梯度流]")
    stats = defaultdict(lambda: {'ok': 0, 'none': 0, 'nan': 0, 'max': 0.0})
    for name, p in actual.named_parameters():
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
        val = v.item() if isinstance(v, torch.Tensor) else float(v)
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


# =========================================================================
# 细粒度Monkey-patch计时
# =========================================================================

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

    # Track整体
    _wrap(actual, 'forward_track_train', '  forward_track_train')

    # 图像特征: backbone + neck 细分
    _wrap(actual, 'extract_img_feat', '    extract_img_feat')
    if hasattr(actual, 'grid_mask'):
        _wrap(actual, 'grid_mask', '      gridmask')
    if hasattr(actual, 'img_backbone'):
        _wrap(actual.img_backbone, 'forward', '      img_backbone')
    if hasattr(actual, 'img_neck'):
        _wrap(actual.img_neck, 'forward', '      img_neck')

    # 点云特征: voxelize + backbone 细分
    _wrap(actual, 'extract_pts_feat', '    extract_pts_feat')
    if hasattr(actual, 'pts_voxel_layer'):
        _wrap(actual, 'voxelize', '      voxelize')
    if hasattr(actual, 'pts_backbone') and actual.pts_backbone is not None:
        _wrap(actual.pts_backbone, 'forward', '      pts_backbone')

    # BEV encoder + detection decoder
    _wrap(actual.pts_bbox_head, 'get_bev_features', '    get_bev_features')
    _wrap(actual.pts_bbox_head, 'get_detections', '    get_detections')

    # 匹配 + 记忆 + QIM
    if hasattr(actual, 'criterion'):
        _wrap(actual.criterion, 'match_for_single_frame', '    match_single_frame')
    if hasattr(actual, 'memory_bank'):
        _wrap(actual.memory_bank, 'forward', '    memory_bank')
    if hasattr(actual, 'query_interact'):
        _wrap(actual.query_interact, 'forward', '    query_interact')

    # Seg head
    if actual.with_seg_head:
        _wrap(actual.seg_head, 'forward_train', '  seg_head.forward_train')

    # forward_train 整体
    orig_ft = actual.forward_train
    def timed_ft(*a, **kw):
        timer.start('forward_total')
        r = orig_ft(*a, **kw)
        timer.end('forward_total')
        return r
    actual.forward_train = timed_ft


# =========================================================================
# Config覆盖 (mock模式, 缩小模型适应24GB)
# =========================================================================

def override_cfg_for_mock(cfg):
    small_bev_h, small_bev_w = 50, 50
    cfg.queue_length = 1

    head = cfg.model.pts_bbox_head
    head.bev_h = small_bev_h
    head.bev_w = small_bev_w
    head.transformer.encoder.num_layers = 2
    head.transformer.decoder.num_layers = 2
    head.positional_encoding.row_num_embed = small_bev_h
    head.positional_encoding.col_num_embed = small_bev_w

    if hasattr(cfg.model, 'seg_head') and cfg.model.seg_head is not None:
        seg = cfg.model.seg_head
        seg.bev_h = small_bev_h
        seg.bev_w = small_bev_w
        seg.canvas_size = (small_bev_h, small_bev_w)
        if hasattr(seg, 'transformer'):
            seg.transformer.encoder.num_layers = 2
            seg.transformer.decoder.num_layers = 2

    for head_name in ['motion_head', 'occ_head', 'planning_head']:
        h = getattr(cfg.model, head_name, None)
        if h is not None and isinstance(h, dict) and 'bev_size' in h:
            h['bev_size'] = (small_bev_h, small_bev_w)

    return cfg


# =========================================================================
# Mock数据生成器
# =========================================================================

def mock_data_iterator(cfg, device, n_iters=5):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes

    queue_length = cfg.get('queue_length', 1)
    past_steps = cfg.get('past_steps', 4)
    fut_steps = cfg.get('fut_steps', 4)
    traj_steps = past_steps + fut_steps
    bev_h = cfg.model.pts_bbox_head.get('bev_h', 50)
    bev_w = cfg.model.pts_bbox_head.get('bev_w', 50)
    num_cams, H, W, num_gt, num_pts = 6, 128, 192, 3, 5000

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

        data = {
            'img': torch.randn(1, queue_length, num_cams, 3, H, W, device=device),
            'img_metas': [metas],
            'points': [[torch.randn(num_pts, 5, device=device)
                         for _ in range(queue_length)]],
        }

        gt_boxes, gt_labels, gt_inds = [], [], []
        gt_past_traj, gt_past_traj_mask = [], []
        gt_sdc_bbox, gt_sdc_label = [], []

        for fi in range(queue_length):
            gt_boxes.append(LiDARInstance3DBoxes(
                torch.randn(num_gt, 9).numpy(), box_dim=9))
            gt_labels.append(torch.randint(0, 10, (num_gt,), device=device))
            gt_inds.append(torch.arange(1, num_gt + 1, device=device))
            gt_past_traj.append(
                torch.randn(num_gt, traj_steps, 2, device=device))
            gt_past_traj_mask.append(
                torch.ones(num_gt, traj_steps, 2, device=device))
            gt_sdc_bbox.append(LiDARInstance3DBoxes(
                torch.randn(1, 9).numpy(), box_dim=9))
            gt_sdc_label.append(
                torch.zeros(1, dtype=torch.long, device=device))

        data.update({
            'gt_bboxes_3d': [gt_boxes], 'gt_labels_3d': [gt_labels],
            'gt_inds': [gt_inds],
            'gt_past_traj': [gt_past_traj],
            'gt_past_traj_mask': [gt_past_traj_mask],
            'gt_sdc_bbox': [gt_sdc_bbox], 'gt_sdc_label': [gt_sdc_label],
            'l2g_r_mat': [[torch.eye(3, device=device)
                           for _ in range(queue_length)]],
            'l2g_t': [[torch.zeros(3, device=device)
                       for _ in range(queue_length)]],
            'timestamp': [[torch.tensor(fi * 0.5, device=device)
                           for fi in range(queue_length)]],
            'gt_fut_traj': [torch.randn(num_gt, 12, 2, device=device)],
            'gt_fut_traj_mask': [torch.ones(num_gt, 12, device=device)],
            'gt_sdc_fut_traj': [torch.randn(1, 12, 2, device=device)],
            'gt_sdc_fut_traj_mask': [torch.ones(1, 12, device=device)],
            'gt_lane_labels': [torch.randint(0, 4, (20,), device=device)],
            'gt_lane_bboxes': [torch.randn(20, 4, device=device)],
            'gt_lane_masks': [torch.randint(
                0, 2, (20, bev_h, bev_w), device=device).float()],
            'gt_segmentation': None, 'gt_instance': None,
            'gt_occ_img_is_valid': None,
            'sdc_planning': None, 'sdc_planning_mask': None,
            'command': None, 'gt_future_boxes': None,
        })
        yield data


# =========================================================================
# 主入口
# =========================================================================

def main():
    args = parse_args()

    # --- 分布式初始化 ---
    distributed, rank, world_size = setup_dist(args.launcher)
    local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'

    log = lambda msg: log_rank0(msg, rank)

    # --- Config ---
    from mmcv import Config
    cfg = Config.fromfile(args.config)
    log(f"[Config] {args.config}")
    log(f"  model={cfg.model.type}, queue={cfg.get('queue_length','?')}, "
        f"epochs={cfg.get('total_epochs','?')}, "
        f"lr={cfg.optimizer.lr}, bs={cfg.data.samples_per_gpu}")
    log(f"  distributed={distributed}, world_size={world_size}")

    # --- Plugin ---
    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        _module_path = os.path.dirname(cfg.plugin_dir).replace('/', '.')
        importlib.import_module(_module_path)

    # --- Mock模式: 缩小模型 ---
    if args.no_data:
        cfg = override_cfg_for_mock(cfg)
        log(f"  [Mock] bev=50x50, encoder=2L, queue=1")

    # --- 构建模型 ---
    from mmdet3d.models import build_model
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    if distributed:
        from mmcv.parallel import MMDistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', True))
        log(f"[Model] DDP, {world_size} GPUs")
    else:
        model = model.to(device)
    model.train()

    if rank == 0:
        actual = model.module if hasattr(model, 'module') else model
        analyze_model_structure(actual)
        check_freeze_status(actual)

    # --- 优化器 ---
    from mmcv.runner import build_optimizer
    optimizer = build_optimizer(
        model.module if hasattr(model, 'module') else model,
        cfg.optimizer)
    log(f"[Optimizer] {cfg.optimizer.type}, "
        f"{len(optimizer.param_groups)} param groups")

    # --- 数据 ---
    timer = StageTimer(enabled=args.profile and rank == 0)

    if args.no_data:
        log("[Data] Mock模式")
        data_iter = mock_data_iterator(cfg, device, n_iters=args.debug_iters)
    else:
        from mmdet3d.datasets import build_dataset
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader

        cfg.data.train.is_debug = True
        cfg.data.train.len_debug = args.debug_samples
        log(f"[Data] is_debug=True, len_debug={args.debug_samples}")

        dataset = build_dataset(cfg.data.train)
        log(f"[Data] 数据集大小: {len(dataset)}")

        dataloader = build_dataloader(
            dataset,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            num_gpus=world_size,
            dist=distributed,
            seed=args.seed,
            shuffler_sampler=cfg.data.get(
                'shuffler_sampler', dict(type='DistributedGroupSampler')),
            nonshuffler_sampler=cfg.data.get(
                'nonshuffler_sampler', dict(type='DistributedSampler')),
        )
        data_iter = iter(dataloader)
        log(f"[Data] DataLoader OK, workers={cfg.data.workers_per_gpu}")

    # --- 注入计时 ---
    if args.profile and rank == 0:
        patch_timing(model, timer)

    # --- 训练循环 ---
    log(f"\n[Train] 开始 {args.debug_iters} iters")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for it in range(args.debug_iters):
        timer.start('data_loading')
        try:
            data = next(data_iter)
        except StopIteration:
            log(f"  iter {it}: 数据耗尽")
            break
        timer.end('data_loading')

        timer.start('iter_total')

        # 前向
        try:
            if distributed:
                losses = model(return_loss=True, **data)
            else:
                losses = model.forward(return_loss=True, **data)
        except Exception as e:
            log(f"  iter {it}: 前向失败 - {e}")
            import traceback; traceback.print_exc()
            break

        if rank == 0:
            check_loss_values(losses, it)

        # 反向
        timer.start('backward')
        optimizer.zero_grad()
        total_loss = sum(v for v in losses.values()
                         if isinstance(v, torch.Tensor) and v.requires_grad)
        try:
            total_loss.backward()
        except Exception as e:
            log(f"  iter {it}: 反向失败 - {e}")
            import traceback; traceback.print_exc()
            break
        timer.end('backward')

        # 梯度裁剪
        timer.start('grad_clip')
        grad_clip = cfg.optimizer_config.get('grad_clip', None)
        if grad_clip:
            params = (model.module if hasattr(model, 'module')
                      else model).parameters()
            gn = torch.nn.utils.clip_grad_norm_(params, **grad_clip)
            if rank == 0:
                print(f"  grad_norm={gn:.2f} (clip={grad_clip['max_norm']})")
        timer.end('grad_clip')

        # 梯度检查 (仅第一个iter)
        if args.check_grad and it == 0 and rank == 0:
            check_gradient_flow(model)

        # 更新
        timer.start('optimizer_step')
        optimizer.step()
        timer.end('optimizer_step')

        timer.end('iter_total')

        if rank == 0 and torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated(device) / 1024**3
            log(f"  iter {it}: loss={total_loss.item():.4f}, "
                f"peak_mem={mem:.2f}GB")

    # --- 汇总 ---
    if rank == 0:
        if args.profile:
            timer.report(
                f"FusionAD Stage1 耗时 ({args.debug_iters} iters, "
                f"{world_size} GPUs)")
        if torch.cuda.is_available():
            log(f"\nGPU {local_rank} 峰值显存: "
                f"{torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
        log("Done.")


if __name__ == '__main__':
    main()
