from ml_prepare.constants import IMAGE_RESOLUTION
from ml_prepare.refuge.utils import base_rgb, base_gray, RefugeConfig
from ml_prepare.refuge.tfds import Refuge


def get_refuge_builder(resolution=IMAGE_RESOLUTION, rgb=True, data_dir=None):
    if resolution is None:
        config = base_rgb if rgb else base_gray
    else:
        config = RefugeConfig(resolution, rgb)
    return Refuge(config=config, data_dir=data_dir)


__all__ = ['get_refuge_builder']
