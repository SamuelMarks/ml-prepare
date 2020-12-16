"""bmes dataset."""
import os
import posixpath
from tempfile import mkdtemp

import tensorflow as tf
import tensorflow_datasets as tfds

# TODO(bmes): Markdown description  that will appear on the catalog page.
import ml_prepare.bmes.get_data
from ml_prepare.constants import IMAGE_RESOLUTION
from ml_prepare.datasets import datasets2classes

_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(bmes): BibTeX citation
_CITATION = """
"""


class Bmes(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for bmes dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    resolution = IMAGE_RESOLUTION
    rgb = True

    def __init__(self, *, dataset_name, data_dir, config, version):
        self.dataset_name = dataset_name
        super(Bmes, self).__init__(data_dir=data_dir, config=config, version=version)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(bmes): Specifies the tfds.core.DatasetInfo object
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
                        num_classes=datasets2classes["bmes"]
                    ),
                    "image/filename": tfds.features.Text(),
                }
            ),
            supervised_keys=("image", "label"),
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(bmes): Downloads the data and defines the splits
        directory = ml_prepare.bmes.get_data.get_data(
            os.path.join(
                os.path.expanduser("~"), "/tensorflow_datasets", self.dataset_name
            )
        )

        print("directory:", directory, ";")

        # TODO(bmes): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(directory),
            "test": self._generate_examples(directory),
            "valid": self._generate_examples(directory),
        }

    def _generate_examples(self, label_images):
        """Generate example for each image in the dict."""

        temp_dir = mkdtemp(prefix=self.dataset_name)
        for label, image_paths in label_images.items():
            for image_path in image_paths:
                key = posixpath.sep.join((label, posixpath.basename(image_path)))

                temp_image_filename = os.path.join(
                    temp_dir,
                    key.replace(posixpath.sep, "_").replace(os.path.sep, "_"),
                )

                if Bmes.session._closed:
                    Bmes.session = tf.compat.v1.Session()
                    Bmes.session.__enter__()

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
                result = Bmes.session.run(fwrite)

                yield key, {
                    "image/filename": temp_image_filename,
                    "image": temp_image_filename,
                    "label": label,
                }

        print("resolved all files, now you should delete: {!r}".format(temp_dir))
        if not Bmes.session._closed:
            Bmes.session.__exit__(None, None, None)
