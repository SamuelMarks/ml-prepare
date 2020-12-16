"""dr_spoc dataset."""

import tensorflow_datasets as tfds

# TODO(dr_spoc): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(dr_spoc): BibTeX citation
_CITATION = """
"""


class DrSpoc(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dr_spoc dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(dr_spoc): Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # e.g. ('image', 'label')
            homepage="https://dataset-homepage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(dr_spoc): Downloads the data and defines the splits
        path = dl_manager.download_and_extract("https://todo-data-url")

        # TODO(dr_spoc): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(dr_spoc): Yields (key, example) tuples from the dataset
        yield "key", {}
