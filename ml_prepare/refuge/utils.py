from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow_datasets.public_api as tfds

from ml_prepare._tfds import transformer


class RefugeTask(object):
    CLASSIFICATION = 'classification'
    SEGMENTATION = 'segmentation'
    LOCALIZATION = 'localization'

    @classmethod
    def all(cls):
        return (
            RefugeTask.CLASSIFICATION,
            RefugeTask.SEGMENTATION,
            RefugeTask.LOCALIZATION,
        )

    @classmethod
    def validate(cls, task):
        if task not in cls.all():
            raise ValueError('Invalid task \'{:s}\': must be one of {:s}'.format(task, str(cls.all())))


def _load_fovea(archive, subpath):
    import xlrd
    # I struggled to get openpyxl to read already opened zip files
    # necessary to tfds to work with google cloud buckets etc
    # could always decompress the entire directory, but that will result in
    # a much bigger disk space usage
    with archive.open(subpath) as fp:
        wb = xlrd.open_workbook(file_contents=fp.read())
        sheet = wb.sheet_by_index(0)
        data = {}
        for i in range(sheet.ncols):
            col_data = sheet.col(i)
            data[col_data[0].value] = [v.value for v in col_data[1:]]
    return data


def _seg_to_label(seg):
    out = np.zeros(shape=seg.shape[:2] + (1,), dtype=np.uint8)
    out[seg == 128] = 1
    out[seg == 0] = 2
    return out


def _load_image(image_fp):
    return np.array(tfds.core.lazy_imports.PIL_Image.open(image_fp))


def RefugeConfig(resolution, rgb=True):
    return transformer.ImageTransformerConfig(
        description='Refuge grand-challenge dataset', resolution=resolution,
        rgb=rgb)


base_rgb = RefugeConfig(None)
base_gray = RefugeConfig(None, rgb=False)

__all__ = ['RefugeTask']
