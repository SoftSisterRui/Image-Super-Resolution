# Copyright (c) OpenMMLab. All rights reserved.
import sys

from mmedit.utils import modify_args


def test_modify_args():
    sys.argv = ['test.py', '--arg_1=1', '--arg-2=2']
    modify_args()
