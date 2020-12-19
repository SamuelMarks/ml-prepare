"""bmes dataset."""

from ml_prepare._tfds import BaseImageLabelFolder


class Bmes(BaseImageLabelFolder):
    _name = "bmes"

    def __init__(self, *, data_dir: str):
        super(Bmes, self).__init__(data_dir=data_dir, dataset_name=self._name)


del BaseImageLabelFolder

__all__ = ["Bmes"]
