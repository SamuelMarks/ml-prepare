# !/usr/bin/env python

from argparse import ArgumentParser, Action, ArgumentTypeError
from enum import Enum
from itertools import chain
from os import path, listdir

import tensorflow_datasets as tfds

from ml_prepare import __version__
from ml_prepare import dr_spoc
from ml_prepare.constants import IMAGE_RESOLUTION
from ml_prepare.dr_spoc.tfds import dr_spoc_datasets, dr_spoc_datasets_set, dr_spoc_builder
from ml_prepare.utils import camel_case


# Originally from https://stackoverflow.com/a/60750535
class EnumAction(Action):
    """
    Argparse action for handling Enums
    """

    def __init__(self, **kwargs):
        # Pop off the type value
        enum = kwargs.pop("type", None)

        # Ensure an Enum subclass is provided
        if enum is None:
            raise ValueError("type must be assigned an Enum when using EnumAction")
        if not issubclass(enum, Enum):
            raise TypeError("type must be an Enum when using EnumAction")

        # Generate choices from the Enum
        kwargs.setdefault("choices", tuple(e.value for e in enum))

        super(EnumAction, self).__init__(**kwargs)

        self._enum = enum

    def __call__(self, parser, namespace, values, option_string=None):
        # Convert value back into an Enum
        enum = self._enum(values)
        setattr(namespace, self.dest, enum)


class ChannelAction(Action):
    """
    Argparse action for handling channels
    """

    def __call__(self, parser, namespace, values, option_string=None):
        if values.isdigit():
            values = int(values)
            if values not in (1, 3):
                parser.error('channels must be `1` or `3`')
            values = (None, 'grayscale', None, 'rgb')[values]
        setattr(namespace, self.dest, values)


# Originally from https://stackoverflow.com/a/33181083; then edited here
class PathType(object):
    def __init__(self, exists=True, type='file', dash_ok=True):
        """

        :param exists:
                True: a path that does exist
                False: a path that does not exist, in a valid parent directory
                None: don't care
        :type exists: bool or None

        :param type: file, dir, symlink, None, or a function returning True for valid paths
                     None: don't care
        :type type: str or bool or None or (() -> bool or None) or ((str) -> bool or None)

        :param dash_ok: whether to allow "-" as stdin/stdout
        :type dash_ok: bool
        """

        assert exists in (True, False, None)
        assert type in ('file', 'dir', 'symlink', None) or hasattr(type, '__call__')

        self._exists = exists
        self._type = type
        self._dash_ok = dash_ok

    def __call__(self, string):
        """

        :param string:
        :type string: str
        """
        if string == '-':
            # the special argument "-" means sys.std{in,out}
            if self._type == 'dir':
                raise ArgumentTypeError('standard input/output (-) not allowed as directory path')
            elif self._type == 'symlink':
                raise ArgumentTypeError('standard input/output (-) not allowed as symlink path')
            elif not self._dash_ok:
                raise ArgumentTypeError('standard input/output (-) not allowed')
        else:
            e = path.exists(string)
            if self._exists:
                if not e:
                    raise ArgumentTypeError("path does not exist: '{}'".format(string))

                if self._type is None:
                    pass
                elif self._type == 'file':
                    if not path.isfile(string):
                        raise ArgumentTypeError("path is not a file: '{}'".format(string))
                elif self._type == 'symlink':
                    if not path.islink(string):
                        raise ArgumentTypeError("path is not a symlink: '{}'".format(string))
                elif self._type == 'dir':
                    if not path.isdir(string):
                        raise ArgumentTypeError("path is not a directory: '{}'".format(string))
                else:
                    raise ArgumentTypeError("path not valid: '{}'".format(string))
            else:
                if not self._exists and e:
                    raise ArgumentTypeError("path exists: '{}'".format(string))

                p = path.dirname(path.normpath(string)) or '.'
                if not path.isdir(p):
                    raise ArgumentTypeError("parent path is not a directory: '{}'".format(p))
                elif not path.exists(p):
                    raise ArgumentTypeError("parent directory does not exist: '{}'".format(p))


def _build_parser():
    directory = path.dirname(__file__)
    parser = ArgumentParser(
        prog='python -m ml_prepare',
        description='Prepare your datasets for ingestion into ML pipelines.'
    )
    parser.add_argument('--version', action='version', version='%(prog)s {}'.format(__version__))

    parser.add_argument('--dataset',
                        type=Enum('DatasetEnum', tuple(
                            map(lambda p: (camel_case(p, upper=True), p),
                                chain.from_iterable((
                                    filter(lambda p: path.isfile(path.join(directory, p, '__init__.py')),
                                           listdir(directory)),
                                    dr_spoc_datasets[:-1]
                                ))))),
                        action=EnumAction, required=True)

    parser.add_argument('--retrieve', help='Retrieve from this directory (or bucket).',
                        # type=PathType(exists=True, type='dir')
                        )

    parser.add_argument('--generate', help='Generate dataset here. '
                                           'Shuffle and then symbolically link to this directory.',
                        # type=PathType(exists=True, type='dir')
                        )

    parser.add_argument('--tfds', help='Construct TFrecords and other metadata needed for TensorFlow Datasets, '
                                       'to this directory.',
                        # type=PathType(exists=True, type='dir')
                        )

    parser.add_argument('--image-height', help='Image height', type=int, default=IMAGE_RESOLUTION[0])
    parser.add_argument('--image-width', help='Image width', type=int, default=IMAGE_RESOLUTION[1])
    parser.add_argument('--image-channels',
                        help='3 or \'rgb\' for red|green|blue (RGB); 1 or \'grayscale\' for grayscale',
                        default='rgb', action=ChannelAction, choices=('grayscale', '1', 'rgb', '3'))

    return parser


if __name__ == '__main__':
    _parser = _build_parser()
    args = _parser.parse_args()
    print('args:', args, ';')

    if not args.retrieve or not args.generate or not args.tfds:
        _parser.error('tfds, generate, and/or retrieve must be specified')

    if args.dataset.value in dr_spoc_datasets_set:
        if args.generate:
            # Create directory structure filled with symlinked files:
            #
            # |-- dr_spoc
            # |   |-- test
            # |   |   |-- No gradable image
            # |   |   |-- non-referable
            # |   |   `-- referable
            # |   |-- train
            # |   |   |-- No gradable image
            # |   |   |-- non-referable
            # |   |   `-- referable
            # |   `-- valid
            # |       |-- No gradable image
            # |       |-- non-referable
            # |       `-- referable
            # |-- dr_spoc_grad_and_no_grad
            # |   |-- test
            # |   |   |-- No gradable image
            # |   |   `-- gradable
            # |   |-- train
            # |   |   |-- No gradable image
            # |   |   `-- gradable
            # |   `-- valid
            # |       |-- No gradable image
            # |       `-- gradable
            # `-- dr_spoc_no_no_grad
            #     |-- test
            #     |   |-- non-referable
            #     |   `-- referable
            #     |-- train
            #     |   |-- non-referable
            #     |   `-- referable
            #     `-- valid
            #         |-- non-referable
            #         `-- referable
            dr_spoc.get_data(args.retrieve, args.generate)

        builder_factory, data_dir, manual_dir = dr_spoc_builder(dataset_name=args.dataset.value,
                                                                data_dir=args.tfds,
                                                                manual_dir=args.generate,
                                                                dr_spoc_parent_dir=args.generate,
                                                                dr_spoc_init=False)
        print('data_dir:'.ljust(20), '{!r}\n'.format(data_dir),
              'manual_dir:'.ljust(20), '{!r}\n'.format(manual_dir),
              sep='')

        builder = builder_factory(resolution=(args.image_height, args.image_width),
                                  rgb=args.image_channels,
                                  data_dir=data_dir)

        builder.download_and_prepare(
            download_config=tfds.download.DownloadConfig(
                extract_dir=args.tfds,
                manual_dir=manual_dir,
                download_mode=tfds.core.dataset_builder.REUSE_DATASET_IF_EXISTS
            ),
            download_dir=args.tfds
        )
