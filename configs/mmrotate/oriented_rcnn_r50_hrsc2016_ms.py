# Oriented R-CNN R50-FPN (le90) on HRSC2016-MS, trained on the train split only.
#
# MMRotate's DOTA recipe retargeted to our split; nothing is tuned to the YOLO recipe.
# Augmentation is MMRotate's own RandomFlip (horizontal / vertical / diagonal), no rotation,
# no scale jitter. 36 epochs (the 12-epoch DOTA schedule is too short for 603 images),
# SGD lr 0.005 with steps at 24 and 33. No validation during training: the val split is
# the evaluation set. Images are .bmp, which needs DOTADataset's img_suffix (MMRotate dev-1.x).
#
# Run from the repository root after tools/yolo_obb_to_dota_annfiles.py:
#   python <mmrotate>/tools/train.py configs/mmrotate/oriented_rcnn_r50_hrsc2016_ms.py
# '_base_' resolves through an installed MMRotate dev-1.x (pip install -e .); with a plain
# source checkout, use the path <mmrotate>/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py.

_base_ = 'mmrotate::oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'

classes = ('ship',)
data = 'data/mmrotate/hrsc/'

model = dict(roi_head=dict(bbox_head=dict(num_classes=len(classes))))

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data,
        ann_file='annfiles/train/',
        data_prefix=dict(img_path='images/train/'),
        img_suffix='bmp',
        metainfo=dict(classes=classes)))

# no validation or test loop; tools/export_views_mmrotate.py runs inference with the test pipeline
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=100)
param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=36, by_epoch=True, milestones=[24, 33], gamma=0.1),
]
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=12, max_keep_ckpts=2),
                     logger=dict(type='LoggerHook', interval=50))
work_dir = 'work_dirs/mmrotate/oriented_rcnn_r50_hrsc2016_ms'
randomness = dict(seed=3101, deterministic=False)
