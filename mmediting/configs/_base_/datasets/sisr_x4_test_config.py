test_pipeline = [
    dict(
        type='LoadImageFromFile',
        key='img',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(type='PackEditInputs')
]


# test config for DIV2K
div2k_data_root = 'data/DIV2K/'
div2k_dataloader = dict(
    num_workers=4,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='BasicImageDataset',
        ann_file='test_GT.txt',
        metainfo=dict(dataset_type='div2k', task_name='sisr'),
        data_root=div2k_data_root,
        data_prefix=dict(
            img='DIV2K_test_LR/X4', gt='DIV2K_test_HR'),
        pipeline=test_pipeline))
div2k_evaluator = dict(
    type='EditEvaluator',
    metrics=[
        dict(type='PSNR', crop_border=4, prefix='DIV2K'),
        dict(type='SSIM', crop_border=4, prefix='DIV2K'),
    ])

# test config
test_cfg = dict(type='EditTestLoop')
test_dataloader = [
    div2k_dataloader,
]
test_evaluator = [
    div2k_evaluator,
]
