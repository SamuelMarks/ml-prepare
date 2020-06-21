#!/usr/bin/env python

from os import path

from ml_prepare.dr_spoc.utils import main


def get_data(root_directory, manual_dir):
    """

    :param root_directory:
    :type root_directory: str

    :param manual_dir:
    :type manual_dir: str or None

    :return:
    :rtype: ```str```
    """
    directory, df, filename2cat, combined_df = main(root_directory=root_directory,
                                                    manual_dir=manual_dir)

    return directory  # combined_df


if __name__ == '__main__':
    get_data(root_directory=path.join(path.expanduser('~'),
                                      'OneDrive - The University of Sydney (Students)'),
             manual_dir=None)

__all__ = ['get_data']
