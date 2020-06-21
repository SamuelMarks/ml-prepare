from itertools import chain
from os import path, listdir

from ml_prepare.dr_spoc.datasets import dr_spoc_datasets
from ml_prepare.utils import camel_case

root_directory = path.dirname(__file__)

datasets = tuple(
    map(lambda p: (camel_case(p, upper=True), p),
        chain.from_iterable((
            filter(lambda p: not p.startswith('_') and path.isfile(path.join(root_directory, p, '__init__.py')),
                   listdir(root_directory)),
            dr_spoc_datasets[:-1]
        )))
)
datasets2classes = {dataset[1]: 3 if dataset[1] == 'dr_spoc' else 2
                    for dataset in datasets}
