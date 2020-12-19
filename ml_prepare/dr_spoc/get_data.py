#!/usr/bin/env python
from os import listdir, path

from ml_prepare.dr_spoc.utils import main


def get_data(root_directory, manual_dir, name):
    """

    :param root_directory:
    :type root_directory: str

    :param manual_dir:
    :type manual_dir: str or None

    :param name: Name of dataset
    :type name: ```Literal["dr_spoc", "dr_spoc_grad_and_no_grad", "dr_spoc_no_no_grad"]```

    :return: Namedtuple with keys from split to location of split in filesystem
    :rtype: ```Datasets```
    """
    directory, df, filename2cat, combined_df = main(
        root_directory=root_directory, manual_dir=manual_dir
    )
    split_parent = path.abspath(path.join(manual_dir, name))
    from ml_prepare.datasets import Datasets

    return Datasets(
        *map(lambda split: path.join(split_parent, split), listdir(split_parent))
    )


if __name__ == "__main__":
    get_data(
        root_directory=path.join(
            path.expanduser("~"), "OneDrive - The University of Sydney (Students)"
        ),
        manual_dir=None,
    )

__all__ = ["get_data"]
