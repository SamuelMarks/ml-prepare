from tensorflow_datasets import public_api as tfds

from ml_prepare._tfds.base import base_builder
from ml_prepare.bmes import get_data as bmes_get_data
from ml_prepare.constants import IMAGE_RESOLUTION
from ml_prepare.dr_spoc import get_data as dr_spoc_get_data
from ml_prepare.dr_spoc.datasets import dr_spoc_datasets_set
from ml_prepare.refuge import get_refuge_builder


def build_tfds_dataset(
    dataset_name,
    tfds_dir,
    generate_dir,
    retrieve_dir,
    manual_dir=None,
    image_channels=3,
    image_height=IMAGE_RESOLUTION[0],
    image_width=IMAGE_RESOLUTION[1],
):
    """

    :param dataset_name: Name of dataset
    :type dataset_name: str

    :param tfds_dir:
    :type tfds_dir: str

    :param generate_dir:
    :type generate_dir: str

    :param retrieve_dir:
    :type retrieve_dir: str

    :param manual_dir:
    :type manual_dir: str

    :param image_channels:
    :type image_channels: str or int

    :param image_height:
    :type image_height: int

    :param image_width:
    :type image_width: int

    :rtype: tfds.core.DatasetBuilder
    """
    data_builder = builder(
        dataset_name,
        generate_dir,
        image_channels,
        image_height,
        image_width,
        retrieve_dir,
        tfds_dir,
    )
    download_and_prepare_kwargs = None
    if hasattr(data_builder, "download_and_prepare_kwargs"):
        download_and_prepare_kwargs = getattr(
            data_builder, "download_and_prepare_kwargs"
        )
        delattr(data_builder, "download_and_prepare_kwargs")
        if hasattr(data_builder, "manual_dir"):
            delattr(data_builder, "manual_dir")
    elif manual_dir is None and hasattr(data_builder, "manual_dir"):
        manual_dir = getattr(data_builder, "manual_dir")
        delattr(data_builder, "manual_dir")
        if download_and_prepare_kwargs is None:
            download_and_prepare_kwargs = dict(
                download_config=tfds.download.DownloadConfig(
                    extract_dir=tfds_dir,
                    manual_dir=manual_dir,
                    download_mode=tfds.core.dataset_builder.REUSE_DATASET_IF_EXISTS,
                ),
                download_dir=tfds_dir,
            )
    assert download_and_prepare_kwargs is not None

    if isinstance(data_builder, tfds.folder_dataset.ImageFolder):
        ds_all_supervised = data_builder.as_dataset(as_supervised=True)
        print("data_builder.info.splits:", data_builder.info.splits, ';')
    else:
        data_builder.download_and_prepare(**download_and_prepare_kwargs)
    return data_builder


def builder(
    dataset_name,
    generate_dir,
    image_channels,
    image_height,
    image_width,
    retrieve_dir,
    tfds_dir,
):
    """

    :param dataset_name:
    :type dataset_name: str

    :param tfds_dir:
    :type tfds_dir: str

    :param generate_dir:
    :type generate_dir: str

    :param retrieve_dir:
    :type retrieve_dir: str

    :param image_channels:
    :type image_channels: str or int

    :param image_height:
    :type image_height: int

    :param image_width:
    :type image_width: int

    :rtype: tfds.core.DatasetBuilder
    """
    builder_factory = None
    if dataset_name in dr_spoc_datasets_set:
        get_data = dr_spoc_get_data
    elif dataset_name == "bmes":
        get_data = bmes_get_data
    elif dataset_name == "refuge":
        builder_factory = get_refuge_builder
    else:
        raise NotImplementedError(dataset_name)
    if builder_factory is None:
        # noinspection PyUnboundLocalVariable
        builder_factory, data_dir, manual_dir = base_builder(
            data_dir=tfds_dir,
            init=generate_dir,
            manual_dir=generate_dir,
            parent_dir=retrieve_dir,
            force_create=False,
            dataset_name=dataset_name,
            get_data=get_data,
        )
    else:
        data_dir, manual_dir = generate_dir, tfds_dir
    data_builder = builder_factory(
        resolution=(image_height, image_width),
        rgb={"rgb": True, 3: True, "3": True}.get(image_channels, False),
        data_dir=data_dir,
    )
    setattr(data_builder, "manual_dir", manual_dir)
    setattr(
        data_builder,
        "download_and_prepare_kwargs",
        dict(
            download_config=tfds.download.DownloadConfig(
                extract_dir=tfds_dir,
                manual_dir=manual_dir,
                download_mode=tfds.core.dataset_builder.REUSE_DATASET_IF_EXISTS,
            ),
            download_dir=tfds_dir,
        ),
    )
    return data_builder
