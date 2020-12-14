from ml_prepare.constants import IMAGE_RESOLUTION
from ml_prepare.refuge.tfds import Refuge
from ml_prepare.refuge.utils import RefugeConfig, base_gray, base_rgb


def get_refuge_builder(resolution=IMAGE_RESOLUTION, rgb=True, data_dir=None):
    if resolution is None:
        config = base_rgb if rgb else base_gray
    else:
        config = RefugeConfig(resolution, rgb)
    return Refuge(config=config, data_dir=data_dir)


__all__ = ["get_refuge_builder"]
