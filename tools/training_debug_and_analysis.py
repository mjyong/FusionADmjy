#!/usr/bin/env python
"""
=============================================================================
FusionAD Stage1 (Track+Map) 训练流程 Debug & 耗时统计
=============================================================================

完整调用链路 (按 tools/train.py 风格重写):

  tools/train.py main()
    │
    ├── 1. Config.fromfile(config_path)                      # 解析配置
    ├── 2. importlib.import_module(plugin_dir)               # 注册自定义模块
    ├── 3. build_model(cfg.model) → FusionAD.__init__()      # 构建模型
    │       ├── FusionADTrack.__init__() (父类)
    │       │     ├── MVXTwoStageDetector.__init__()
    │       │     │     ├── build img_backbone (ResNet101, frozen_stages=4)
    │       │     │     ├── build img_neck (FPN)
    │       │     │     ├── build pts_voxel_layer (HardVoxelization)
    │       │     │     ├── build pts_backbone (SparseEncoderHD)
    │       │     │     └── build pts_bbox_head (BEVFormerTrackHead)
    │       │     ├── GridMask
    │       │     ├── query_embedding (Embedding 901×512)
    │       │     ├── reference_points (Linear 256→3)
    │       │     ├── QueryInteractionModule
    │       │     ├── MemoryBank
    │       │     ├── ClipMatcher (loss)
    │       │     └── freeze_img_modules → img_backbone.eval() + img_neck.eval()
    │       ├── build seg_head (PansegformerHead)
    │       └── (Stage1没有 motion_head / occ_head / planning_head)
    │
    ├── 4. model.init_weights()                              # 初始化权重
    ├── 5. build_dataset(cfg.data.train) → NuScenesE2EDataset
    │
    └── 6. custom_train_model() → custom_train_detector()    # 启动训练
          │
          ├── build_dataloader()                             # 构建数据加载器
          ├── MMDistributedDataParallel(model)               # DDP包裹
          ├── build_optimizer(AdamW, lr=1.2e-4)              # 优化器
          ├── build_runner(EpochBasedRunner, 20 epochs)      # Runner
          ├── register hooks (lr, optimizer, ckpt, log)
          │
          └── runner.run(data_loaders, workflow)             # 开始训练循环
                │
                ├── [每个epoch]
                │   └── [每个iter]
                │       │
                │       ├── ★ DataLoader.__next__()
                │       │   └── NuScenesE2EDataset.prepare_train_data(index)
                │       │       ├── 取 queue_length=5 连续帧 (同一scene)
                │       │       ├── 对每帧: get_data_info → pipeline
                │       │       │   ├── LoadPointsFromFile (加载LiDAR)
                │       │       │   ├── LoadPointsFromMultiSweeps (10帧sweep)
                │       │       │   ├── LoadMultiViewImageFromFilesInCeph (6路图像)
                │       │       │   ├── PhotoMetricDistortionMultiViewImage
                │       │       │   ├── LoadAnnotations3D_E2E (GT标注)
                │       │       │   ├── GenerateOccFlowLabels (占用流标签)
                │       │       │   ├── ObjectRangeFilterTrack
                │       │       │   ├── PointsRangeFilter + PointShuffle
                │       │       │   ├── NormalizeMultiviewImage
                │       │       │   ├── PadMultiViewImage (size_divisor=32)
                │       │       │   ├── DefaultFormatBundle3D
                │       │       │   └── CustomCollect3D
                │       │       └── union2one(): 拼接5帧→单个样本dict
                │       │           ├── img: [5, 6, 3, H, W]
                │       │           ├── points: list of 5帧点云
                │       │           ├── gt_bboxes_3d: 5帧的GT框
                │       │           ├── can_bus: 相对位姿增量
                │       │           └── ...其他GT标注
                │       │
                │       ├── ★ FusionAD.forward(return_loss=True)
                │       │   └── FusionAD.forward_train(**data)
                │       │       │
                │       │       ├── [Stage1: freeze_track=False, 训练Track]
                │       │       │
                │       │       ├── A. forward_track_train(points, img, ...)
                │       │       │   │   遍历 num_frame=5 帧:
                │       │       │   │
                │       │       │   ├── frame 0..3: _forward_single()
                │       │       │   │   ├── a1. get_bevs()
                │       │       │   │   │   ├── get_history_bev(prev_frames) [no_grad]
                │       │       │   │   │   │   ├── extract_img_feat() → backbone+neck
                │       │       │   │   │   │   └── pts_bbox_head.get_bev_features()
                │       │       │   │   │   ├── extract_feat(current_img)
                │       │       │   │   │   │   └── [freeze_img_modules=True → no_grad]
                │       │       │   │   │   │       ├── GridMask
                │       │       │   │   │   │       ├── img_backbone(ResNet101) → 多尺度特征
                │       │       │   │   │   │       └── img_neck(FPN) → 4级特征图
                │       │       │   │   │   ├── extract_pts_feat(points)
                │       │       │   │   │   │   ├── voxelize() [no_grad]
                │       │       │   │   │   │   │   └── pts_voxel_layer (HardVoxelization)
                │       │       │   │   │   │   └── pts_backbone (SparseEncoderHD)
                │       │       │   │   │   └── pts_bbox_head.get_bev_features()
                │       │       │   │   │       └── PerceptionTransformer.get_bev_features()
                │       │       │   │   │           └── BEVFormerEncoder (6层)
                │       │       │   │   │               └── BEVFormerFusionLayer ×6
                │       │       │   │   │                   ├── PtsCrossAttention (LiDAR→BEV)
                │       │       │   │   │                   ├── SpatialCrossAttention (Cam→BEV)
                │       │       │   │   │                   │   └── MSDeformableAttention3D
                │       │       │   │   │                   ├── TemporalSelfAttention
                │       │       │   │   │                   └── FFN
                │       │       │   │   │   输出: bev_embed [40000, 1, 256], bev_pos
                │       │       │   │   │
                │       │       │   │   ├── a2. pts_bbox_head.get_detections()
                │       │       │   │   │   └── DetectionTransformerDecoder (6层)
                │       │       │   │   │       ├── Self-Attention (query间)
                │       │       │   │   │       ├── Cross-Attention (query→BEV)
                │       │       │   │   │       └── FFN
                │       │       │   │   │   输出: cls_scores, bbox_preds, past_traj_preds
                │       │       │   │   │
                │       │       │   │   ├── a3. criterion.match_for_single_frame()
                │       │       │   │   │   └── HungarianAssigner3DTrack (匈牙利匹配)
                │       │       │   │   │
                │       │       │   │   ├── a4. memory_bank(track_instances) (时序记忆)
                │       │       │   │   └── a5. query_interact() (QIM, query更新/丢弃)
                │       │       │   │
                │       │       │   └── frame 4 (最后一帧): 同上 + 收集输出
                │       │       │       └── losses = criterion.losses_dict
                │       │       │           ├── loss_ce (分类, FocalLoss)
                │       │       │           ├── loss_bbox (回归, L1Loss)
                │       │       │           └── loss_past_traj (历史轨迹)
                │       │       │
                │       │       │   losses_track → loss_weighted_and_prefixed(prefix='track')
                │       │       │
                │       │       ├── B. seg_head.forward_train(bev_embed, ...)
                │       │       │   [Stage1: freeze_seg=True(默认) → 有的config设为False]
                │       │       │   └── PansegformerHead
                │       │       │       ├── SegDeformableTransformer
                │       │       │       │   ├── Encoder (6层 MultiScaleDeformableAttention)
                │       │       │       │   └── Decoder (6层 DeformableDetrTransformerDecoder)
                │       │       │       ├── SegMaskHead (thing) + SegMaskHead (stuff)
                │       │       │       └── losses:
                │       │       │           ├── loss_cls (地图分类)
                │       │       │           ├── loss_bbox (地图框)
                │       │       │           ├── loss_iou (GIoU)
                │       │       │           └── loss_mask (DiceLoss)
                │       │       │
                │       │       │   losses_seg → loss_weighted_and_prefixed(prefix='map')
                │       │       │
                │       │       ├── C. [Stage1没有 motion_head] → 跳过
                │       │       ├── D. [Stage1没有 occ_head] → 跳过
                │       │       └── E. [Stage1没有 planning_head] → 跳过
                │       │
                │       │   return losses = {track.xxx, map.xxx}
                │       │
                │       ├── ★ OptimizerHook.after_train_iter()
                │       │   ├── total_loss = sum(losses.values())
                │       │   ├── total_loss.backward()                   # 反向传播
                │       │   ├── clip_grad_norm_(max_norm=35)            # 梯度裁剪
                │       │   └── optimizer.step() + zero_grad()          # 参数更新
                │       │
                │       └── ★ Hooks:
                │           ├── LrUpdaterHook (CosineAnnealing + linear warmup)
                │           ├── CheckpointHook (每1 epoch保存)
                │           ├── TextLoggerHook (每50 iter打印)
                │           └── TensorboardLoggerHook
                │
                └── [完成20 epochs]

Stage1 训练配置:
  - 模型: FusionAD (仅Track+Map head)
  - 输入: Camera (6view) + LiDAR (10sweep)
  - 冻结: img_backbone(ResNet101) + img_neck(FPN) → eval() + no_grad
  - 可训练: pts_backbone, BEV encoder, Track head, Seg head
  - 优化器: AdamW, lr=1.2e-4, img_backbone lr_mult=0.1
  - 调度: CosineAnnealing, warmup_iters=1
  - queue_length=5, epochs=20, samples_per_gpu=1
  - grad_clip: max_norm=35
"""

from __future__ import division
import argparse
import copy
import json
import logging
import os
import os.path as osp
import sys
import time
import warnings
from collections import defaultdict, OrderedDict

import torch
import torch.nn as nn

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# =============================================================================
# Part 1: 按 tools/train.py 风格的完整训练入口 (含debug插桩)
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='FusionAD Stage1 训练Debug工具')
    parser.add_argument(
        'config',
        nargs='?',
        default='projects/configs/stage1_track_map/fusion_base_track_map.py',
        help='config文件路径')
    parser.add_argument('--work-dir', default='./work_dirs/debug_stage1')
    parser.add_argument('--debug-iters', type=int, default=5,
                        help='只跑N个iter用于debug')
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
# Part 2: 耗时统计器 (对应调用链中每个节点)
# =============================================================================

class StageTimer:
    """精确统计forward_train中每个子阶段的GPU耗时"""

    def __init__(self, enabled=True, log_interval=1):
        self.enabled = enabled
        self.log_interval = log_interval
        self.timings = defaultdict(list)  # stage_name -> [elapsed_ms, ...]
        self._starts = {}
        self.iter_count = 0

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name):
        if not self.enabled:
            return
        self._sync()
        self._starts[name] = time.time()

    def end(self, name):
        if not self.enabled:
            return
        self._sync()
        elapsed = (time.time() - self._starts[name]) * 1000  # ms
        self.timings[name].append(elapsed)

    def report(self, title="耗时统计"):
        if not self.enabled or not self.timings:
            return
        print(f"\n{'='*70}")
        print(f" {title}")
        print(f"{'='*70}")

        # 按预定义顺序输出
        order = [
            'data_loading',
            'forward_total',
            '  forward_track_train',
            '    extract_img_feat',
            '    extract_pts_feat',
            '    get_bev_features',
            '    get_detections',
            '    match+qim+membank',
            '  seg_head.forward_train',
            'backward',
            'optimizer_step',
            'iter_total',
        ]
        # 补充未在order中的key
        all_keys = list(self.timings.keys())
        for k in all_keys:
            if k not in order:
                order.append(k)

        total_avg = 0
        rows = []
        for name in order:
            if name not in self.timings:
                continue
            vals = self.timings[name]
            avg = sum(vals) / len(vals)
            rows.append((name, avg, min(vals), max(vals), len(vals)))
            if name == 'iter_total':
                total_avg = avg

        for name, avg, mn, mx, cnt in rows:
            pct = avg / max(total_avg, 1e-9) * 100 if total_avg > 0 else 0
            indent_name = name
            print(f"  {indent_name:35s}  avg={avg:8.1f}ms  "
                  f"min={mn:7.1f}  max={mx:7.1f}  "
                  f"({pct:5.1f}%)  [{cnt}次]")

        print(f"{'='*70}")

    def reset(self):
        self.timings.clear()
        self._starts.clear()
        self.iter_count = 0


# =============================================================================
# Part 3: 参数/冻结状态分析 (对应 __init__ 阶段)
# =============================================================================

def analyze_model_structure(model, title="模型结构分析"):
    """打印模型各模块参数量、冻结状态、train/eval模式"""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

    total_params = 0
    trainable_params = 0

    for name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
        mod_frozen = mod_total - mod_train
        mode = 'TRAIN' if module.training else 'EVAL'

        total_params += mod_total
        trainable_params += mod_train

        pct = mod_train / max(mod_total, 1) * 100
        print(f"  {name:30s}  "
              f"total={mod_total:>12,d}  "
              f"trainable={mod_train:>12,d}  "
              f"frozen={mod_frozen:>12,d}  "
              f"({pct:5.1f}% trainable)  [{mode}]")

    frozen_params = total_params - trainable_params
    print(f"\n  {'TOTAL':30s}  "
          f"total={total_params:>12,d}  "
          f"trainable={trainable_params:>12,d}  "
          f"frozen={frozen_params:>12,d}  "
          f"({trainable_params/max(total_params,1)*100:.1f}% trainable)")
    print(f"  显存估算: total={total_params*4/1024**3:.2f}GB  "
          f"trainable={trainable_params*4/1024**3:.2f}GB (仅参数, 不含激活/梯度)")
    print(f"{'='*80}")

    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
    }


def check_freeze_status(model):
    """检查FusionAD Stage1的冻结配置是否正确"""
    print("\n[冻结状态检查]")
    checks = []

    # img_backbone 应该冻结
    if hasattr(model, 'img_backbone'):
        bb_trainable = sum(1 for p in model.img_backbone.parameters() if p.requires_grad)
        bb_total = sum(1 for p in model.img_backbone.parameters())
        is_eval = not model.img_backbone.training
        ok = bb_trainable == 0 or is_eval
        status = "OK" if ok else "WARN"
        checks.append(f"  [{status}] img_backbone: {bb_trainable}/{bb_total} trainable, "
                       f"mode={'eval' if is_eval else 'train'}")
        if hasattr(model, 'freeze_img_modules') and model.freeze_img_modules:
            checks.append(f"        freeze_img_modules=True → backbone应为eval+no_grad")

    # img_neck 应该冻结
    if hasattr(model, 'img_neck'):
        nk_trainable = sum(1 for p in model.img_neck.parameters() if p.requires_grad)
        nk_total = sum(1 for p in model.img_neck.parameters())
        is_eval = not model.img_neck.training
        ok = nk_trainable == 0 or is_eval
        status = "OK" if ok else "WARN"
        checks.append(f"  [{status}] img_neck: {nk_trainable}/{nk_total} trainable, "
                       f"mode={'eval' if is_eval else 'train'}")

    # pts_backbone 应该可训练
    if hasattr(model, 'pts_backbone') and model.pts_backbone is not None:
        pt_trainable = sum(1 for p in model.pts_backbone.parameters() if p.requires_grad)
        pt_total = sum(1 for p in model.pts_backbone.parameters())
        ok = pt_trainable == pt_total
        status = "OK" if ok else "WARN"
        checks.append(f"  [{status}] pts_backbone: {pt_trainable}/{pt_total} trainable (应全部可训练)")

    # BEV encoder (pts_bbox_head) 应该可训练 (Stage1)
    if hasattr(model, 'pts_bbox_head'):
        bev_trainable = sum(1 for p in model.pts_bbox_head.parameters() if p.requires_grad)
        bev_total = sum(1 for p in model.pts_bbox_head.parameters())
        ok = bev_trainable > 0
        status = "OK" if ok else "WARN"
        checks.append(f"  [{status}] pts_bbox_head (BEV+Track): {bev_trainable}/{bev_total} trainable")

    # seg_head 应该可训练 (Stage1)
    if hasattr(model, 'seg_head') and model.seg_head is not None:
        seg_trainable = sum(1 for p in model.seg_head.parameters() if p.requires_grad)
        seg_total = sum(1 for p in model.seg_head.parameters())
        ok = seg_trainable > 0
        status = "OK" if ok else "WARN"
        checks.append(f"  [{status}] seg_head (Map): {seg_trainable}/{seg_total} trainable")

    # motion/occ/planning 应该不存在 (Stage1)
    for head_name in ['motion_head', 'occ_head', 'planning_head']:
        has_head = hasattr(model, head_name) and getattr(model, head_name) is not None
        status = "WARN" if has_head else "OK"
        checks.append(f"  [{status}] {head_name}: {'存在(Stage1不应有)' if has_head else '不存在(正确)'}")

    # freeze_track / freeze_seg 标志
    if hasattr(model, 'freeze_track'):
        checks.append(f"  [INFO] freeze_track={model.freeze_track} "
                       f"{'(Stage1应为False)' if model.freeze_track else '(正确)'}")
    if hasattr(model, 'freeze_seg'):
        checks.append(f"  [INFO] freeze_seg={model.freeze_seg}")

    for c in checks:
        print(c)


# =============================================================================
# Part 4: 梯度流检查 (对应 backward 阶段)
# =============================================================================

def check_gradient_flow(model):
    """backward后检查梯度是否正常流动"""
    print("\n[梯度流检查]")

    module_stats = defaultdict(lambda: {
        'has_grad': 0, 'no_grad': 0, 'zero_grad': 0,
        'nan': 0, 'inf': 0, 'max_norm': 0.0,
    })

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_name = name.split('.')[0]
        stats = module_stats[module_name]

        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            stats['has_grad'] += 1
            stats['max_norm'] = max(stats['max_norm'], grad_norm)
            if grad_norm < 1e-10:
                stats['zero_grad'] += 1
            if torch.isnan(param.grad).any():
                stats['nan'] += 1
            if torch.isinf(param.grad).any():
                stats['inf'] += 1
        else:
            stats['no_grad'] += 1

    print(f"  {'模块':25s}  {'有梯度':>6s}  {'无梯度':>6s}  {'零梯度':>6s}  "
          f"{'NaN':>4s}  {'Inf':>4s}  {'最大范数':>12s}  状态")
    print(f"  {'-'*90}")

    for mod in sorted(module_stats.keys()):
        s = module_stats[mod]
        status = "OK"
        if s['nan'] > 0:
            status = "NaN!"
        elif s['inf'] > 0:
            status = "Inf!"
        elif s['no_grad'] > 0 and s['has_grad'] == 0:
            status = "无梯度!"
        elif s['zero_grad'] > s['has_grad'] * 0.5:
            status = "大量零梯度"

        print(f"  {mod:25s}  {s['has_grad']:>6d}  {s['no_grad']:>6d}  {s['zero_grad']:>6d}  "
              f"{s['nan']:>4d}  {s['inf']:>4d}  {s['max_norm']:>12.4e}  {status}")


def check_loss_values(losses, iter_idx):
    """检查loss值是否正常"""
    print(f"\n[Loss检查] iter={iter_idx}")
    task_sums = defaultdict(float)
    warnings_list = []

    for k in sorted(losses.keys()):
        v = losses[k]
        val = v.item() if isinstance(v, torch.Tensor) else v
        prefix = k.split('.')[0]
        task_sums[prefix] += val

        flag = ""
        if val != val:
            flag = " ← NaN!"
            warnings_list.append(f"NaN: {k}")
        elif abs(val) > 1e5:
            flag = " ← 极大值!"
            warnings_list.append(f"极大值: {k}={val}")
        elif abs(val) == float('inf'):
            flag = " ← Inf!"
            warnings_list.append(f"Inf: {k}")
        print(f"    {k:45s} = {val:>12.6f}{flag}")

    print(f"\n  [按任务汇总]")
    total = 0
    for task in sorted(task_sums.keys()):
        print(f"    {task:15s}: {task_sums[task]:>12.6f}")
        total += task_sums[task]
    print(f"    {'TOTAL':15s}: {total:>12.6f}")

    if warnings_list:
        print(f"\n  [WARNINGS]")
        for w in warnings_list:
            print(f"    ⚠ {w}")

    return total


# =============================================================================
# Part 5: Monkey-patch forward_train — 注入子阶段计时
# =============================================================================

def patch_fusionad_stage1_timing(model, timer: StageTimer):
    """
    深度插桩FusionAD的forward调用链, 精确统计:
      - extract_img_feat (backbone+neck, 通常no_grad)
      - extract_pts_feat (voxelize + SparseEncoder)
      - get_bev_features (BEVFormer encoder)
      - get_detections (decoder)
      - match+qim+membank (匹配+query交互+记忆库)
      - seg_head
    """
    actual = model.module if hasattr(model, 'module') else model

    # --- 1. 插桩 extract_img_feat ---
    orig_extract_img = actual.extract_img_feat
    def timed_extract_img(*a, **kw):
        timer.start('    extract_img_feat')
        r = orig_extract_img(*a, **kw)
        timer.end('    extract_img_feat')
        return r
    actual.extract_img_feat = timed_extract_img

    # --- 2. 插桩 extract_pts_feat ---
    orig_extract_pts = actual.extract_pts_feat
    def timed_extract_pts(*a, **kw):
        timer.start('    extract_pts_feat')
        r = orig_extract_pts(*a, **kw)
        timer.end('    extract_pts_feat')
        return r
    actual.extract_pts_feat = timed_extract_pts

    # --- 3. 插桩 pts_bbox_head.get_bev_features ---
    orig_get_bev = actual.pts_bbox_head.get_bev_features
    def timed_get_bev(*a, **kw):
        timer.start('    get_bev_features')
        r = orig_get_bev(*a, **kw)
        timer.end('    get_bev_features')
        return r
    actual.pts_bbox_head.get_bev_features = timed_get_bev

    # --- 4. 插桩 pts_bbox_head.get_detections ---
    orig_get_det = actual.pts_bbox_head.get_detections
    def timed_get_det(*a, **kw):
        timer.start('    get_detections')
        r = orig_get_det(*a, **kw)
        timer.end('    get_detections')
        return r
    actual.pts_bbox_head.get_detections = timed_get_det

    # --- 5. 插桩 forward_track_train (整体) ---
    orig_ftt = actual.forward_track_train
    def timed_ftt(*a, **kw):
        timer.start('  forward_track_train')
        r = orig_ftt(*a, **kw)
        timer.end('  forward_track_train')
        return r
    actual.forward_track_train = timed_ftt

    # --- 6. 插桩 seg_head.forward_train ---
    if actual.with_seg_head:
        orig_seg = actual.seg_head.forward_train
        def timed_seg(*a, **kw):
            timer.start('  seg_head.forward_train')
            r = orig_seg(*a, **kw)
            timer.end('  seg_head.forward_train')
            return r
        actual.seg_head.forward_train = timed_seg

    # --- 7. 插桩 forward_train (整体) ---
    orig_ft = actual.forward_train
    def timed_ft(*a, **kw):
        timer.start('forward_total')
        r = orig_ft(*a, **kw)
        timer.end('forward_total')
        return r
    actual.forward_train = timed_ft

    logger.info("[Timer] 已插桩FusionAD Stage1 forward调用链")


# =============================================================================
# Part 6: 主入口 — 按 tools/train.py 的流程, 加debug逻辑
# =============================================================================

def main():
    args = parse_args()

    # ============================
    # Step 1: 解析配置 (对应 train.py L106)
    # ============================
    print("\n" + "#" * 80)
    print("# Step 1: 解析配置")
    print("#" * 80)

    try:
        from mmcv import Config
    except ImportError:
        print("[ERROR] mmcv未安装, 请: pip install mmcv-full")
        print("以下打印调用链文档供参考:")
        print(__doc__)
        return

    cfg = Config.fromfile(args.config)
    print(f"  配置文件: {args.config}")
    print(f"  模型类型: {cfg.model.type}")
    print(f"  queue_length: {cfg.get('queue_length', 'N/A')}")
    print(f"  total_epochs: {cfg.get('total_epochs', 'N/A')}")
    print(f"  optimizer: {cfg.optimizer.type}, lr={cfg.optimizer.lr}")
    print(f"  samples_per_gpu: {cfg.data.samples_per_gpu}")
    print(f"  freeze_img_modules: {cfg.model.get('freeze_img_modules', False)}")
    print(f"  freeze_bev_encoder: {cfg.model.get('freeze_bev_encoder', False)}")

    # ============================
    # Step 2: 注册Plugin (对应 train.py L115-136)
    # ============================
    print("\n" + "#" * 80)
    print("# Step 2: 注册Plugin")
    print("#" * 80)

    if hasattr(cfg, 'plugin') and cfg.plugin:
        import importlib
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_path = _module_dir.replace('/', '.')
        logger.info(f"  加载plugin: {_module_path}")
        try:
            plg_lib = importlib.import_module(_module_path)
            print(f"  [OK] Plugin加载成功: {_module_path}")
        except Exception as e:
            print(f"  [ERROR] Plugin加载失败: {e}")
            return

    # ============================
    # Step 3: 构建模型 (对应 train.py L216-220)
    # ============================
    print("\n" + "#" * 80)
    print("# Step 3: 构建模型 FusionAD")
    print("#" * 80)

    from mmdet3d.models import build_model
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    try:
        model = build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        model = model.to(device)
        model.train()
        print(f"  [OK] 模型构建成功, 设备={device}")
    except Exception as e:
        print(f"  [ERROR] 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ============================
    # Step 3.5: 分析模型结构和冻结状态
    # ============================
    analyze_model_structure(model, "FusionAD Stage1 模型结构")
    check_freeze_status(model)

    # ============================
    # Step 4: 构建优化器 (对应 mmdet_train.py L91)
    # ============================
    print("\n" + "#" * 80)
    print("# Step 4: 构建优化器")
    print("#" * 80)

    from mmcv.runner import build_optimizer
    optimizer = build_optimizer(model, cfg.optimizer)

    # 打印各参数组的lr
    for i, pg in enumerate(optimizer.param_groups):
        num_params = sum(p.numel() for p in pg['params'])
        print(f"  参数组{i}: lr={pg['lr']:.6f}, "
              f"weight_decay={pg.get('weight_decay', 0)}, "
              f"参数量={num_params:,d}")

    # ============================
    # Step 5: 构建数据 / Mock数据
    # ============================
    print("\n" + "#" * 80)
    print("# Step 5: 准备训练数据")
    print("#" * 80)

    timer = StageTimer(enabled=args.profile, log_interval=1)

    if args.no_data:
        print("  [Mock模式] 使用伪造数据")
        data_iter = mock_data_iterator(cfg, device, n_iters=args.debug_iters)
    else:
        try:
            from mmdet3d.datasets import build_dataset
            from projects.mmdet3d_plugin.datasets.builder import build_dataloader
            dataset = build_dataset(cfg.data.train)
            print(f"  数据集大小: {len(dataset)}")
            dataloader = build_dataloader(
                dataset,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                len([args.gpu_id]),
                dist=False,
                seed=args.seed,
                shuffler_sampler=cfg.data.get('shuffler_sampler', dict(type='DistributedGroupSampler')),
                nonshuffler_sampler=cfg.data.get('nonshuffler_sampler', dict(type='DistributedSampler')),
            )
            data_iter = iter(dataloader)
            print(f"  [OK] DataLoader构建成功, batch_size={cfg.data.samples_per_gpu}")
        except Exception as e:
            print(f"  [WARN] 数据集加载失败: {e}")
            print(f"  [FALLBACK] 使用Mock数据")
            data_iter = mock_data_iterator(cfg, device, n_iters=args.debug_iters)

    # ============================
    # Step 5.5: 注入计时器
    # ============================
    if args.profile:
        patch_fusionad_stage1_timing(model, timer)

    # ============================
    # Step 6: 训练循环 (对应 runner.run)
    # ============================
    print("\n" + "#" * 80)
    print(f"# Step 6: 训练循环 (共{args.debug_iters}个iter)")
    print("#" * 80)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for iter_idx in range(args.debug_iters):
        print(f"\n{'─'*60}")
        print(f"  Iter {iter_idx + 1}/{args.debug_iters}")
        print(f"{'─'*60}")

        # --- 数据加载 ---
        timer.start('data_loading')
        try:
            data = next(data_iter)
        except StopIteration:
            print("  [WARN] 数据迭代器耗尽, 结束")
            break
        timer.end('data_loading')

        # --- 前向传播 ---
        timer.start('iter_total')
        try:
            losses = model.forward(return_loss=True, **data)
            print(f"  [OK] 前向传播成功")
        except Exception as e:
            print(f"  [FAIL] 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            break

        # --- 检查Loss ---
        total_loss_val = check_loss_values(losses, iter_idx)

        # --- 反向传播 ---
        timer.start('backward')
        optimizer.zero_grad()
        total_loss = sum(v for v in losses.values()
                         if isinstance(v, torch.Tensor) and v.requires_grad)
        try:
            total_loss.backward()
            print(f"\n  [OK] 反向传播成功, total_loss={total_loss.item():.6f}")
        except Exception as e:
            print(f"  [FAIL] 反向传播失败: {e}")
            import traceback
            traceback.print_exc()
            break
        timer.end('backward')

        # --- 梯度裁剪 ---
        grad_clip = cfg.optimizer_config.get('grad_clip', None)
        if grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), **grad_clip)
            print(f"  梯度范数(裁剪前): {grad_norm:.4f}, max_norm={grad_clip['max_norm']}")

        # --- 检查梯度 ---
        if args.check_grad:
            check_gradient_flow(model)

        # --- 优化器更新 ---
        timer.start('optimizer_step')
        optimizer.step()
        timer.end('optimizer_step')

        timer.end('iter_total')
        print(f"  [OK] 参数更新完成")

    # ============================
    # Step 7: 汇总报告
    # ============================
    print("\n" + "#" * 80)
    print("# Step 7: 训练Debug汇总")
    print("#" * 80)

    if args.profile:
        timer.report(f"FusionAD Stage1 耗时统计 ({args.debug_iters} iters)")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"\n  GPU {args.gpu_id} 峰值显存: {peak_mem:.2f} GB")

    print("\n  训练调用链路详见本文件顶部的文档字符串")
    print("  完成!")


# =============================================================================
# Part 7: Mock数据生成器 (不需要真实数据集)
# =============================================================================

def mock_data_iterator(cfg, device, n_iters=5):
    """
    生成与NuScenesE2EDataset.prepare_train_data → union2one
    输出格式一致的mock数据。

    对应调用链:
      DataLoader → prepare_train_data() → pipeline → union2one()
      输出的data dict包含:
        img: [B, queue_length, 6, 3, H, W]
        points: list[list[Tensor]]  (外层batch, 内层queue帧)
        img_metas: list[dict{0..queue_length-1}]
        gt_bboxes_3d, gt_labels_3d, gt_inds, ... (每帧一个列表)
    """
    import numpy as np

    queue_length = cfg.get('queue_length', 5)
    num_cams = 6
    H, W = 928, 1600
    num_gt = 8
    pc_range = cfg.model.get('pc_range', [-54, -54, -5, 54, 54, 3])
    bev_h = cfg.model.pts_bbox_head.get('bev_h', 200)
    bev_w = cfg.model.pts_bbox_head.get('bev_w', 200)

    for i in range(n_iters):
        # img_metas: mmcv的DataContainer解包后是 list[dict]
        # 在collate后: img_metas = [dict{0: meta_frame0, 1: meta_frame1, ...}]
        metas = {}
        for fi in range(queue_length):
            metas[fi] = {
                'scene_token': 'debug_scene_0',
                'prev_bev': fi > 0,
                'can_bus': np.zeros(18, dtype=np.float64),
                'lidar2img': [np.eye(4, dtype=np.float32) for _ in range(num_cams)],
                'img_shape': [(H, W, 3)] * num_cams,
                'box_type_3d': _get_box_type(pc_range),
            }

        # 构造与collate后格式一致的data
        from mmdet3d.core.bbox import LiDARInstance3DBoxes

        data = {}
        data['img'] = torch.randn(1, queue_length, num_cams, 3, H, W, device=device)
        data['img_metas'] = [metas]

        # points: 需要是 list[list[Tensor]]
        # collate后: points = [DataContainer] → .data = list[Tensor per frame]
        # forward_track_train 中: points[0] 是一个 list of queue_length tensors
        data['points'] = [[torch.randn(30000, 5, device=device) for _ in range(queue_length)]]

        # GT数据: 每项都是 list[list[per_frame]]
        gt_boxes_list = []
        gt_labels_list = []
        gt_inds_list = []
        gt_past_traj_list = []
        gt_past_traj_mask_list = []
        gt_sdc_bbox_list = []
        gt_sdc_label_list = []

        for fi in range(queue_length):
            boxes = torch.randn(num_gt, 9, device=device)
            gt_boxes_list.append(LiDARInstance3DBoxes(boxes.cpu()))
            gt_labels_list.append(torch.randint(0, 10, (num_gt,), device=device))
            gt_inds_list.append(torch.arange(1, num_gt + 1, device=device))
            gt_past_traj_list.append(torch.randn(num_gt, 4, 2, device=device))
            gt_past_traj_mask_list.append(torch.ones(num_gt, 4, device=device))
            sdc_box = torch.randn(1, 9, device=device)
            gt_sdc_bbox_list.append(LiDARInstance3DBoxes(sdc_box.cpu()))
            gt_sdc_label_list.append(torch.zeros(1, dtype=torch.long, device=device))

        data['gt_bboxes_3d'] = [gt_boxes_list]
        data['gt_labels_3d'] = [gt_labels_list]
        data['gt_inds'] = [gt_inds_list]
        data['gt_past_traj'] = [gt_past_traj_list]
        data['gt_past_traj_mask'] = [gt_past_traj_mask_list]
        data['gt_sdc_bbox'] = [gt_sdc_bbox_list]
        data['gt_sdc_label'] = [gt_sdc_label_list]

        data['l2g_r_mat'] = [[torch.eye(3, device=device) for _ in range(queue_length)]]
        data['l2g_t'] = [[torch.zeros(3, device=device) for _ in range(queue_length)]]
        data['timestamp'] = [[torch.tensor(fi * 0.5, device=device) for fi in range(queue_length)]]

        # 最后一帧的GT (轨迹/未来)
        data['gt_fut_traj'] = [torch.randn(num_gt, 12, 2, device=device)]
        data['gt_fut_traj_mask'] = [torch.ones(num_gt, 12, device=device)]
        data['gt_sdc_fut_traj'] = [torch.randn(1, 12, 2, device=device)]
        data['gt_sdc_fut_traj_mask'] = [torch.ones(1, 12, device=device)]

        # Map GT
        data['gt_lane_labels'] = [torch.randint(0, 4, (50,), device=device)]
        data['gt_lane_bboxes'] = [torch.randn(50, 4, device=device)]
        data['gt_lane_masks'] = [torch.randint(0, 2, (50, bev_h, bev_w), device=device).float()]

        # Occ GT (Stage1不需要但forward_train接受)
        data['gt_segmentation'] = None
        data['gt_instance'] = None
        data['gt_occ_img_is_valid'] = None

        # Planning GT (Stage1不需要)
        data['sdc_planning'] = None
        data['sdc_planning_mask'] = None
        data['command'] = None
        data['gt_future_boxes'] = None

        yield data


def _get_box_type(pc_range):
    """获取box_type_3d"""
    try:
        from mmdet3d.core.bbox import LiDARInstance3DBoxes
        return LiDARInstance3DBoxes
    except ImportError:
        return None


if __name__ == '__main__':
    main()
