# Oriented R-CNN R50-FPN (le90) on DIOR-R (20 classes), train split only.
#
# MMRotate's DOTA recipe retargeted: 12 epochs, SGD lr 0.005, RandomFlip h/v/diagonal, no
# rotation, no scale jitter. The 800-pixel images are resized to 1024, the size the YOLO11
# detectors train and test at. No validation during training: the val split is the
# evaluation set. Convert the annotations with --min-side 2 (one zero-width training box).
#
# Run from the repository root after tools/yolo_obb_to_dota_annfiles.py:
#   python <mmrotate>/tools/train.py configs/mmrotate/oriented_rcnn_r50_dior_r.py
# '_base_' resolves through an installed MMRotate dev-1.x (pip install -e .); with a plain
# source checkout, use the path <mmrotate>/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py.

_base_ = 'mmrotate::oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'

# the dataset yaml's order, spaces as hyphens (classes.json of the annotation converter)
classes = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge', 'chimney',
           'dam', 'Expressway-Service-area', 'Expressway-toll-station', 'golffield',
           'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
           'tenniscourt', 'trainstation', 'vehicle', 'windmill')
data = 'data/mmrotate/dior_r/'

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
work_dir = 'work_dirs/mmrotate/oriented_rcnn_r50_dior_r'
randomness = dict(seed=3104, deterministic=False)
