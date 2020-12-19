"""dr_spoc dataset."""

from ml_prepare._tfds import BaseImageLabelFolder


class DrSpoc(BaseImageLabelFolder):
    _name = "dr_spoc"

    def __init__(self, *, data_dir: str):
        super(DrSpoc, self).__init__(data_dir=data_dir, dataset_name=self._name)


class DrSpocNoNoGrad(BaseImageLabelFolder):
    _name = "dr_spoc_no_no_grad"

    def __init__(self, *, data_dir: str):
        super(DrSpocNoNoGrad, self).__init__(data_dir=data_dir, dataset_name=self._name)


class DrSpocGradAndNoGrad(BaseImageLabelFolder):
    _name = "dr_spoc_grad_and_no_grad"

    def __init__(self, *, data_dir: str):
        super(DrSpocGradAndNoGrad, self).__init__(
            data_dir=data_dir, dataset_name=self._name
        )


del BaseImageLabelFolder

__all__ = ["DrSpoc", "DrSpocNoNoGrad", "DrSpocGradAndNoGrad"]
