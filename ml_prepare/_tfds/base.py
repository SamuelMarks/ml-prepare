"""base dataset class."""
import os
import posixpath
from os import path
from tempfile import mkdtemp
from typing import Callable, NamedTuple, Optional, Tuple, Union

try:
    import tensorflow as tf
except ImportError:
    tf = None  # Just used for typing
import tensorflow_datasets.public_api as tfds

import ml_prepare.datasets
from ml_prepare.constants import IMAGE_RESOLUTION

_DESCRIPTION: str = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION: str = """
"""


class BaseImageLabelFolder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    """DatasetBuilder for image datasets."""

    VERSION: tfds.core.Version = tfds.core.Version("1.0.0")
    RELEASE_NOTES: dict = {
        "1.0.0": "Initial release.",
    }

    def __init__(
        self,
        *,
        data_dir: str,
        dataset_name: str,
        rgb: bool = True,
        get_data: Optional[
            Callable[
                [str],
                NamedTuple("Datasets", [("train", str), ("valid", str), ("test", str)]),
            ]
        ] = None,
        retrieve_dir: Optional[str] = None,
        resolution: Tuple[int, int] = IMAGE_RESOLUTION,
        config: Union[None, str, tfds.core.BuilderConfig] = None,
        version: Union[None, str, tfds.core.utils.Version] = None,
        shape: Optional[tfds.core.utils.Shape] = None,
        dtype: Optional[tf.DType] = None,
    ):
        self.name = dataset_name
        if self.name is None:
            raise TypeError("Builders must have a name")
        self.get_data = get_data
        self._image_shape = shape
        self._image_dtype = dtype
        self.resolution = resolution
        self.retrieve_dir = retrieve_dir
        self.rgb = rgb
        super(BaseImageLabelFolder, self).__init__(
            data_dir=data_dir, config=config, version=version
        )

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(  # shape=resolution + ((3 if rgb else 1),),
                        encoding_format="jpeg",
                        shape=self._image_shape,
                        dtype=self._image_dtype,
                    ),
                    "label": tfds.features.ClassLabel(
                        num_classes=ml_prepare.datasets.datasets2classes[self.name]
                    ),
                    "image/filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _description(self) -> dict:
        return {
            "image/filename": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "image": tf.io.FixedLenSequenceFeature(
                shape=self._image_shape, dtype=self._image_dtype
            )
            #'image_raw': tf.io.FixedLenFeature([], tf.string),
        }

    def _split_generators(self, dl_manager: tfds.download.DownloadManager) -> dict:
        """Returns SplitGenerators."""
        # Acquires the data and defines the splits

        if self.get_data is not None:
            return dict(
                map(
                    lambda _split_directory: (
                        _split_directory[0],
                        self._generate_examples(_split_directory[1]),
                    ),
                    self.get_data(self.retrieve_dir)._asdict().items(),
                )
            )

        return {
            "train": self._generate_examples(self.data_dir),
            "test": self._generate_examples(self.data_dir),
            "valid": self._generate_examples(self.data_dir),
        }

    def _generate_examples(self, label_images: Union[str, dict]):
        """Generate example for each image in the dict."""

        temp_dir = mkdtemp(prefix=self.name)

        if isinstance(label_images, str):
            assert path.isdir(label_images)
            print("label_images:", label_images, ";")
            (
                self._split_examples,
                labels,
            ) = tfds.folder_dataset.image_folder._get_split_label_images(
                path.dirname(label_images)
            )
            self.info.features["label"].names = sorted(labels)
            split_dict = tfds.core.SplitDict(self.name)

            label_images = {label: [] for label in self.info.features["label"].names}

            for split_name, examples in self._split_examples.items():
                split_dict.add(
                    tfds.core.SplitInfo(
                        name=split_name,
                        shard_lengths=[len(examples)],
                    )
                )

                # TODO: This in a generator so it doesn't fill memory
                for example in examples:
                    label_images[example.label].append(example.image_path)
            self.info.update_splits_if_different(split_dict)

        for label, image_paths in label_images.items():
            for image_path in image_paths:
                key = posixpath.sep.join((label, posixpath.basename(image_path)))

                temp_image_filename = os.path.join(
                    temp_dir,
                    key.replace(posixpath.sep, "_").replace(os.path.sep, "_"),
                )

                if BaseImageLabelFolder.session._closed:
                    BaseImageLabelFolder.session = tf.compat.v1.Session()
                    BaseImageLabelFolder.session.__enter__()

                image_decoded = tf.image.decode_jpeg(
                    tf.io.read_file(image_path), channels=3 if self.rgb else 1
                )
                resized = tf.image.resize(image_decoded, self.resolution)
                enc = tf.image.encode_jpeg(
                    tf.cast(resized, tf.uint8),
                    "rgb" if self.rgb else "grayscale",
                    quality=100,
                    chroma_downsampling=False,
                )
                fwrite = tf.io.write_file(tf.constant(temp_image_filename), enc)
                result = BaseImageLabelFolder.session.run(fwrite)

                yield key, {
                    "image/filename": temp_image_filename,
                    "image": temp_image_filename,
                    "label": label,
                }

        print("resolved all files, now you should delete: {!r}".format(temp_dir))
        if not BaseImageLabelFolder.session._closed:
            BaseImageLabelFolder.session.__exit__(None, None, None)


BaseImageLabelFolder.session = type("FakeSession", tuple(), {"_closed": True})()


def _get_manual_dir(parent_dir: str, manual_dir: Optional[str]) -> str:
    return (
        (
            manual_dir
            if "symlinked_datasets" in frozenset(manual_dir.split(path.sep))
            else path.join(manual_dir, "symlinked_datasets")
        )
        if manual_dir is not None and path.isabs(manual_dir)
        else path.join(parent_dir, "symlinked_datasets")
    )


__all__ = ["BaseImageLabelFolder"]
