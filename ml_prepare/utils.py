from collections import deque
from functools import reduce
from itertools import islice
from os import makedirs, path
from pprint import PrettyPrinter
from random import sample

import quantumrandom

it_consumes = (
    lambda it, n=None: deque(it, maxlen=0)
    if n is None
    else next(islice(it, n, n), None)
)

pp = PrettyPrinter(indent=4).pprint


def create_random_numbers(minimum, maximum, n):  # type: (int, int, int) -> [int]
    whole, prev = frozenset(), frozenset()
    while len(whole) < n:
        whole = reduce(
            frozenset.union,
            (
                frozenset(
                    map(
                        lambda num: minimum + (num % maximum),
                        quantumrandom.get_data(data_type="uint16", array_length=1024),
                    )
                ),
                prev,
            ),
        )
        prev = whole
        print(len(whole), "of", n)
    return sample(whole, n)


def ensure_is_dir(filepath):  # type: (str) -> str
    assert filepath is not None and path.isdir(
        filepath
    ), "{!r} is not a directory".format(filepath)
    return filepath


def camel_case(st, upper=False):
    output = "".join(x for x in st.title() if x.isalnum())
    return getattr(output[0], "upper" if upper else "lower")() + output[1:]


def update_d(d, arg=None, **kwargs):
    if arg:
        d.update(arg)
    if kwargs:
        d.update(kwargs)
    return d


def run_once(f):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return f(*args, **kwargs)

    wrapper.has_run = False
    return wrapper


def infer_data_dir(directory, dataset_name):
    if path.isdir(directory):
        if not path.basename(directory) == dataset_name:
            _dir = path.join("symlinked_datasets", dataset_name)
            directory = path.normpath(directory)
            if not directory.endswith(_dir):
                fst = path.join(path.dirname(directory), _dir)
                snd = path.join(path.join(path.dirname(directory), "downloads"), _dir)
                directory = next(filter(path.isdir, (fst, snd)))
                # print("made new data_dir of", data_dir)
    else:
        makedirs(directory)

    return directory
