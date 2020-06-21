import tensorflow_datasets as tfds

from ml_prepare._tfds.base import base_builder
from ml_prepare.bmes import get_data as bmes_get_data
from ml_prepare.dr_spoc import get_data as dr_spoc_get_data
from ml_prepare.dr_spoc.datasets import dr_spoc_datasets_set


def build_tfds_dataset(tfds_dir, generate_dir, retrieve_dir, dataset_name,
                       image_channels, image_height, image_width):
    """

    :param tfds_dir:
    :type tfds_dir: str

    :param generate_dir:
    :type generate_dir: str

    :param retrieve_dir:
    :type retrieve_dir: str

    :param dataset_name:
    :type dataset_name: str

    :param image_channels:
    :type image_channels: str or int

    :param image_height:
    :type image_height: int

    :param image_width:
    :type image_width: int
    """
    if dataset_name in dr_spoc_datasets_set:
        get_data = dr_spoc_get_data
    elif dataset_name == 'bmes':
        get_data = bmes_get_data
    else:
        raise NotImplementedError(dataset_name)

    builder_factory, data_dir, manual_dir = base_builder(
        data_dir=tfds_dir, init=generate_dir, manual_dir=generate_dir,
        parent_dir=retrieve_dir, force_create=False, dataset_name=dataset_name,
        get_data=get_data
    )
    builder = builder_factory(resolution=(image_height, image_width),
                              rgb=image_channels, data_dir=data_dir)
    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            extract_dir=tfds_dir,
            manual_dir=manual_dir,
            download_mode=tfds.core.dataset_builder.REUSE_DATASET_IF_EXISTS
        ),
        download_dir=tfds_dir
    )
