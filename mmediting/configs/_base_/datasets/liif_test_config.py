scale_test_list = [2, 3, 4, 6, 18, 30]

test_pipelines = [[
    dict(
        type='LoadImageFromFile',
        key='gt',
        color_type='color',
        channel_order='rgb',
        imdecode_backend='cv2'),
    dict(
        type='RandomDownSampling', scale_min=scale_test, scale_max=scale_test),
    dict(type='GenerateCoordinateAndCell', scale=scale_test, reshape_gt=False),
    dict(type='PackEditInputs')
] for scale_test in scale_test_list]


# test config for DIV2K
div2k_dataloaders = [
    dict(
        num_workers=4,
        persistent_workers=False,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type='BasicImageDataset',
            ann_file='test_GT.txt',
            metainfo=dict(dataset_type='div2k', task_name='sisr'),
            data_root='data/DIV2K',
            data_prefix=dict(
                img='DIV2K_test_LR/X4', gt='DIV2K_test_HR'),
            pipeline=test_pipeline)) for test_pipeline in test_pipelines
]
div2k_evaluators = [[
    dict(type='PSNR', crop_border=scale, prefix=f'DIV2Kx{scale}'),
    dict(type='SSIM', crop_border=scale, prefix=f'DIV2Kx{scale}'),
] for scale in scale_test_list]

# test config
test_cfg = dict(type='EditTestLoop')
test_dataloader = [
    *div2k_dataloaders,
]
test_evaluator = [
    *div2k_evaluators,
]
