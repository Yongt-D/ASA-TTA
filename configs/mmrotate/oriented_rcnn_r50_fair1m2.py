# Oriented R-CNN R50-FPN (le90) on FAIR1M-2.0 (RGB, 37 classes), train split of our split.
#
# MMRotate's DOTA recipe retargeted: 12 epochs, SGD lr 0.005, RandomFlip h/v/diagonal, no
# rotation, no scale jitter. The inherited pipeline resizes the 600-1000 pixel images to fit
# 1024 with the aspect ratio kept. No validation during training: the val split is the
# evaluation set.
#
# Run from the repository root after tools/yolo_obb_to_dota_annfiles.py:
#   python <mmrotate>/tools/train.py configs/mmrotate/oriented_rcnn_r50_fair1m2.py
# '_base_' resolves through an installed MMRotate dev-1.x (pip install -e .); with a plain
# source checkout, use the path <mmrotate>/configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py.

_base_ = 'mmrotate::oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py'

# the dataset yaml's order, spaces as hyphens (classes.json of the annotation converter)
classes = ('A220', 'A321', 'A330', 'A350', 'ARJ21', 'Baseball-Field', 'Basketball-Court',
           'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'Bridge', 'Bus', 'C919',
           'Cargo-Truck', 'Dry-Cargo-Ship', 'Dump-Truck', 'Engineering-Ship', 'Excavator',
           'Fishing-Boat', 'Football-Field', 'Intersection', 'Liquid-Cargo-Ship', 'Motorboat',
           'Passenger-Ship', 'Roundabout', 'Small-Car', 'Tennis-Court', 'Tractor', 'Trailer',
           'Truck-Tractor', 'Tugboat', 'Van', 'Warship', 'other-airplane', 'other-ship',
           'other-vehicle')
data = 'data/mmrotate/fair1m2/'

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
work_dir = 'work_dirs/mmrotate/oriented_rcnn_r50_fair1m2'
randomness = dict(seed=3103, deterministic=False)
