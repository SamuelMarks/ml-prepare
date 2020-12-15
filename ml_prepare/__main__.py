# !/usr/bin/env python

from argparse import ArgumentParser
from enum import Enum

from argparse_utils.actions.channel import ChannelAction
from argparse_utils.actions.enum import EnumAction

from ml_prepare import __version__, datasets
from ml_prepare.constants import IMAGE_RESOLUTION
from ml_prepare.datasets import datasets
from ml_prepare.exectors import build_tfds_dataset


def _build_parser():
    parser = ArgumentParser(
        prog="python -m ml_prepare",
        description="Prepares your datasets for ingestion into ML pipelines.",
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s {}".format(__version__)
    )

    parser.add_argument(
        "--dataset",
        type=Enum("DatasetEnum", datasets),
        action=EnumAction,
        required=True,
    )

    parser.add_argument(
        "--retrieve",
        help="Retrieve from this directory (or bucket).",
        # type=PathType(exists=True, type='dir')
    )

    parser.add_argument(
        "--generate",
        help="Generate dataset here. "
        "Shuffle and then symbolically link to this directory.",
        # type=PathType(exists=True, type='dir')
    )

    parser.add_argument(
        "--tfds",
        help="Construct TFrecords and other metadata needed for TensorFlow Datasets, "
        "to this directory.",
        # type=PathType(exists=True, type='dir')
    )

    parser.add_argument(
        "--image-height", help="Image height", type=int, default=IMAGE_RESOLUTION[0]
    )
    parser.add_argument(
        "--image-width", help="Image width", type=int, default=IMAGE_RESOLUTION[1]
    )
    parser.add_argument(
        "--image-channels",
        help="3 or 'rgb' for red|green|blue (RGB); 1 or 'grayscale' for grayscale",
        default="rgb",
        action=ChannelAction,
        choices=("grayscale", "1", "rgb", "3"),
    )

    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    args = _parser.parse_args()

    if (not args.retrieve and not args.generate) or not args.tfds:
        _parser.error("tfds, generate, and/or retrieve must be specified")

    build_tfds_dataset(
        dataset_name=args.dataset.value,
        generate_dir=args.generate,  # if path.basename(args.generate) == args.dataset.value else path.join(args.generate, args.dataset.value, "symlinked_datasets", args.dataset.value),
        retrieve_dir=args.retrieve,
        tfds_dir=args.tfds,
        image_channels=args.image_channels,
        image_height=args.image_height,
        image_width=args.image_width,
    )
