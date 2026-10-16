# Oriented R-CNN R50-FPN (le90) on our DOTA-v1.0 crops (1024 pixels, gap 200), train split only.
#
# MMRotate's released DOTA weights are trained on train + val, and our evaluation crops come
# from val, so the detector is retrained on train. Schedule, optimiser and augmentation are
# MMRotate's 1x DOTA recipe unchanged (12 epochs, SGD lr 0.005, RandomFlip h/v/diagonal).
# No validation during training: the val split is the evaluation set.
#
# Run from the repository root after tools/yolo_obb_to_dota_annfiles.py:
#   python <mmrotate>/tools/train.py configs/mmrotate/oriented_rcnn_r50_dota.py
# '_base_' resolves through an installed MMRotate dev-1.x (pip install -e .); with a plain
# source checkout, use the path <mmrotate>/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py.

_base_ = 'mmrotate::oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'

# the dataset yaml's order, spaces as hyphens (classes.json of the annotation converter)
classes = ('plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',
           'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',
           'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool')
data = 'data/mmrotate/dota/'

model = dict(roi_head=dict(bbox_head=dict(num_classes=len(classes))))

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data,
        ann_file='annfiles/train/',
        data_prefix=dict(img_path='images/train/'),
        img_suffix='jpg',
        metainfo=dict(classes=classes)))

# no validation or test loop; tools/export_views_mmrotate.py runs inference with the test pipeline
val_dataloader = None
val_evaluator = None
val_cfg = None
test_dataloader = None
test_evaluator = None
test_cfg = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=100)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=4, max_keep_ckpts=2),
                     logger=dict(type='LoggerHook', interval=50))
work_dir = 'work_dirs/mmrotate/oriented_rcnn_r50_dota'
randomness = dict(seed=3102, deterministic=False)
