# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmedit.models.base_archs import PixelShufflePack


def test_pixel_shuffle():

    # test on cpu
    model = PixelShufflePack(3, 3, 2, 3)
    model.init_weights()
    x = torch.rand(1, 3, 16, 16)
    y = model(x)
    assert y.shape == (1, 3, 32, 32)

    # test on gpu
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        y = model(x)
        assert y.shape == (1, 3, 32, 32)
