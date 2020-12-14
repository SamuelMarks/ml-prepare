import os
from os import path
from tempfile import mkdtemp

import tensorflow as tf
import tensorflow_datasets as tfds

from ml_prepare import get_logger
from ml_prepare.datasets import datasets2classes

logger = get_logger(
    ".".join(
        (
            path.basename(path.dirname(__file__)),
            path.basename(__file__).rpartition(".")[0],
        )
    )
)


def base_builder(
    dataset_name, data_dir, init, parent_dir, manual_dir, get_data, force_create=False
):
    """

    :param dataset_name:
    :type dataset_name: ``str``

    :param data_dir:
    :type data_dir: ``str``

    :param init:
    :type init: ``str``

    :param parent_dir:
    :type parent_dir: ``str``

    :param manual_dir:
    :type manual_dir: ``str``

    :param get_data: Function which parses data source and creates symlinks, returning symlink dir
    :type get_data: ``(str,str or None) -> str``

    :param force_create:
    :type force_create: ``bool``

    :return: builder_factory, data_dir, manual_dir
    :rtype: ((int, bool, str) -> (tfds.folder_dataset.ImageFolder)), str, str
    """

    manual_dir = _get_manual_dir(parent_dir, manual_dir)
    if init:
        if manual_dir is None:
            raise ValueError("`manual_dir` must be provided if `init is True`")
        elif parent_dir is None:
            raise ValueError("`parent_dir` must be provided if " "`init is True`")
        elif force_create or not path.isdir(
            path.join(_get_manual_dir(parent_dir, manual_dir), dataset_name)
        ):
            get_data_resp = get_data(parent_dir, manual_dir)
            if (
                get_data_resp is not None
                and hasattr(get_data_resp, "train")
                and isinstance(get_data_resp.train, str)
                and path.isdir(get_data_resp.train)
                and path.basename(get_data_resp.train) != path.basename(manual_dir)
            ):

                maybe_manual_dir = path.dirname(path.dirname(get_data_resp.train))
                if path.basename(manual_dir) != path.basename(maybe_manual_dir):
                    manual_dir = maybe_manual_dir
        else:
            logger.info("Using already created symlinks")

        part = "tensorflow_datasets"
        if path.basename(data_dir) not in frozenset((part, "tfds")):
            data_dir = path.join(data_dir, part)

        assert path.isdir(manual_dir), (
            "Manual directory {!r} does not exist. "
            "Create it and download/extract dataset artifacts "
            "in there. Additional instructions: "
            "This is a 'template' dataset.".format(manual_dir)
        )

    def builder_factory(
        resolution, rgb, data_dir
    ):  # type: (int, bool, str) -> tfds.folder_dataset.ImageFolder
        print("resolution:".ljust(20), "{!r}".format(resolution), sep="")

        class BaseImageLabelFolder(tfds.folder_dataset.ImageFolder):
            def _info(self):
                return tfds.core.DatasetInfo(
                    builder=self,
                    description="TODO",
                    features=tfds.features.FeaturesDict(
                        {
                            "image": tfds.features.Image(  # shape=resolution + ((3 if rgb else 1),),
                                encoding_format="jpeg"
                            ),
                            "label": tfds.features.ClassLabel(
                                num_classes=datasets2classes[dataset_name]
                            ),
                        }
                    ),
                    supervised_keys=("image", "label"),
                )

            def _generate_examples(self, label_images):
                """Generate example for each image in the dict."""

                temp_dir = mkdtemp(prefix=dataset_name)
                for label, image_paths in label_images.items():
                    for image_path in image_paths:
                        key = "/".join((label, os.path.basename(image_path)))

                        temp_image_filename = path.join(
                            temp_dir, key.replace(path.sep, "_")
                        )

                        if base_builder.session._closed:
                            base_builder.session = tf.compat.v1.Session()
                            base_builder.session.__enter__()

                        image_decoded = tf.image.decode_jpeg(
                            tf.io.read_file(image_path), channels=3 if rgb else 1
                        )
                        resized = tf.image.resize(image_decoded, resolution)
                        enc = tf.image.encode_jpeg(
                            tf.cast(resized, tf.uint8),
                            "rgb" if rgb else "grayscale",
                            quality=100,
                            chroma_downsampling=False,
                        )
                        fwrite = tf.io.write_file(tf.constant(temp_image_filename), enc)
                        result = base_builder.session.run(fwrite)

                        yield key, {
                            "image": temp_image_filename,
                            "label": label,
                        }

                print(
                    "resolved all files, now you should delete: {!r}".format(temp_dir)
                )
                if not base_builder.session._closed:
                    base_builder.session.__exit__(None, None, None)

        builder = BaseImageLabelFolder(root_dir=data_dir)
        builder.dataset_name = dataset_name

        return builder

    if manual_dir is not None and not path.basename(manual_dir) == dataset_name:
        manual_dir = path.join(manual_dir, dataset_name)

    return builder_factory, data_dir, manual_dir


base_builder.session = type("FakeSession", tuple(), {"_closed": True})()


def _get_manual_dir(parent_dir, manual_dir):  # type: (str, str or None) -> str
    return (
        (
            manual_dir
            if path.basename(manual_dir) == "symlinked_datasets"
            or "{sep}symlinked_datasets{sep}".format(sep=path.sep) in manual_dir
            else path.join(manual_dir, "symlinked_datasets")
        )
        if manual_dir is not None and path.isabs(manual_dir)
        else path.join(parent_dir, "symlinked_datasets")
    )


__all__ = ["base_builder"]
