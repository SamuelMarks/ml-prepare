import os
import zipfile

import numpy as np
import tensorflow as tf
import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.core.download import DownloadManager

from ml_prepare.refuge.utils import (
    RefugeTask,
    _load_fovea,
    _load_image,
    _seg_to_label,
    base_gray,
    base_rgb,
)


class Refuge(tfds.core.GeneratorBasedBuilder):
    """
    Glaucoma related dataset builder for REFUGE grand challenge.

    We save data for all tasks in the one set of tfrecords for each resolution.
    This -may- come at a very slight performance penalty and result in slightly
    larger files if one is only concerned with the classification task, but
    makes access more convenience and avoids duplicating data on disk for the
    different tasks.
    """

    BUILDER_CONFIGS = [base_rgb, base_gray]

    URL = "http://refuge.grand-challenge.org"

    def __init__(self, task=RefugeTask.CLASSIFICATION, **kwargs):
        RefugeTask.validate(task)
        self.task = task
        super(Refuge, self).__init__(**kwargs)

    def _info(self):
        resolution = self.builder_config.resolution
        num_channels = 3 if self.builder_config.rgb else 1
        if resolution is None:
            h, w = None, None
        else:
            h, w = resolution
        task = self.task
        label_key = {
            RefugeTask.CLASSIFICATION: "label",
            RefugeTask.SEGMENTATION: "segmentation",
            RefugeTask.LOCALIZATION: "macular_center",
        }[task]
        return tfds.core.DatasetInfo(
            builder=self,
            description=self.builder_config.description,
            features=tfds.features.FeaturesDict(
                {
                    "fundus": tfds.features.Image(shape=(h, w, num_channels)),
                    "segmentation": tfds.features.Image(shape=(h, w, 1)),
                    "label": tfds.features.Tensor(dtype=tf.bool, shape=()),
                    "macular_center": tfds.features.Tensor(
                        dtype=tf.float32, shape=(2,)
                    ),
                    "index": tfds.features.Tensor(dtype=tf.int64, shape=()),
                }
            ),
            homepage=self.URL,
            citation="TODO",
            supervised_keys=("fundus", label_key),
        )

    def _split_generators(
        self, dl_manager
    ):  # type: (Refuge, DownloadManager) -> [tfds.core.SplitGenerator]
        base_url = "https://aipe-broad-dataset.cdn.bcebos.com/gno"
        urls_d = {
            "train": {
                "annotations": "Annotation-Training400.zip",
                "fundi": "REFUGE-Training400.zip",
            },
            "validation": {
                "annotations": "REFUGE-Validation400-GT.zip",
                "fundi": "REFUGE-Validation400.zip",
            },
            "test": {"fundi": "REFUGE-Test400.zip"},
        }
        urls = tf.nest.map_structure(  # pylint: disable=no-member
            lambda x: os.path.join(base_url, x), urls_d
        )

        dl_manager._sizes_checksums.update(
            {
                "{base_url}/{filename}".format(base_url=base_url, filename=filename): (
                    size,
                    sha256,
                )
                for filename, (size, sha256) in {
                    urls_d["train"]["annotations"]: (
                        348137626,
                        "5dc4b16c5e4c3f7c30106b0f22f0b44b7d4bdc18f12d3a0d2e0cadc41fd8e92b",
                    ),
                    urls_d["train"]["fundi"]: (
                        716807624,
                        "c6d4f8cd66b0b558f8f63245803cba53e774ab6c5de9dfb7bd113a6e30fb183a",
                    ),
                    urls_d["validation"]["annotations"]: (
                        5320597,
                        "e535a7bf0ad7bc6e94ffa68f8bda5e4965be0972d31c44f610c60beb3e1a9e84",
                    ),
                    urls_d["validation"]["fundi"]: (
                        363541061,
                        "41867a62214bd2da5750bd9c4cbd2f25802f5340a3042732227ebe2286a867c6",
                    ),
                    urls_d["test"]["fundi"]: (
                        360832658,
                        "ff8a2f14ec94812186310cbf490a69c6deed9d9377c1f9dd5ba05ee0ff1a99cb",
                    ),
                }.items()
            }
        )
        download_dirs = dl_manager.download(urls)

        return [
            tfds.core.SplitGenerator(
                name=split, gen_kwargs=dict(split=split, **download_dirs[split])
            )
            for split in ("train", "validation", "test")
        ]

    def _generate_examples(self, split, **kwargs):
        return {
            "train": self._generate_train_examples,
            "validation": self._generate_validation_examples,
            "test": self._generate_test_examples,
        }[split](**kwargs)

    def _generate_train_examples(self, fundi, annotations):
        with tf.io.gfile.GFile(annotations, "rb") as annotations:
            annotations = zipfile.ZipFile(annotations)
            fov_data = _load_fovea(
                annotations,
                os.path.join("Annotation-Training400", "Fovea_location.xlsx"),
            )
            xys = {
                fundus_fn: (x, y)
                for fundus_fn, x, y in zip(
                    fov_data["ImgName"], fov_data["Fovea_X"], fov_data["Fovea_Y"]
                )
            }

            with tf.io.gfile.GFile(fundi, "rb") as fundi:
                fundi = zipfile.ZipFile(fundi)

                def get_example(label, _fundus_path, segmentation_path):
                    fundus_fn = _fundus_path.split("/")[-1]
                    xy = np.array(xys[fundus_fn], dtype=np.float32)
                    fundus = _load_image(fundi.open(_fundus_path))
                    seg = _load_image(annotations.open(segmentation_path))
                    seg = _seg_to_label(seg)
                    image_res = fundus.shape[:2]
                    assert seg.shape[:2] == image_res
                    _transformer = self.builder_config.transformer(image_res)
                    if _transformer is not None:
                        xy = _transformer.transform_point(xy)
                        fundus = _transformer.transform_image(
                            fundus, interp=tf.image.ResizeMethod.BILINEAR
                        )
                        seg = _transformer.transform_image(
                            seg, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                        )
                    return {
                        "fundus": fundus,
                        "segmentation": seg,
                        "label": label,
                        "macular_center": xy,
                        "index": index,
                    }

                # positive data_preparation_scripts
                for index in range(1, 41):
                    fundus_path = os.path.join(
                        "Training400", "Glaucoma", "g{:04d}.jpg".format(index)
                    )
                    seg_path = os.path.join(
                        "Annotation-Training400",
                        "Disc_Cup_Masks",
                        "Glaucoma",
                        "g{:04d}.bmp".format(index),
                    )

                    yield (True, index), get_example(True, fundus_path, seg_path)

                # negative data_preparation_scripts
                for index in range(1, 361):
                    fundus_path = os.path.join(
                        "Training400", "Non-Glaucoma", "n{:04d}.jpg".format(index)
                    )
                    seg_path = os.path.join(
                        "Annotation-Training400",
                        "Disc_Cup_Masks",
                        "Non-Glaucoma",
                        "n{:04d}.bmp".format(index),
                    )
                    yield (False, index), get_example(False, fundus_path, seg_path)

    def _generate_validation_examples(self, fundi, annotations):
        with tf.io.gfile.GFile(annotations, "rb") as annotations:
            annotations = zipfile.ZipFile(annotations)
            fov_data = _load_fovea(
                annotations,
                os.path.join("REFUGE-Validation400-GT", "Fovea_locations.xlsx"),
            )
            label_data = {
                fundus_fn: (x, y, bool(label))
                for fundus_fn, x, y, label in zip(
                    fov_data["ImgName"],
                    fov_data["Fovea_X"],
                    fov_data["Fovea_Y"],
                    fov_data["Glaucoma Label"],
                )
            }

            with tf.io.gfile.GFile(fundi, "rb") as fundi:
                fundi = zipfile.ZipFile(fundi)

                for index in range(1, 401):
                    seg_fn = "V{:04d}.bmp".format(index)
                    seg_path = os.path.join(
                        "REFUGE-Validation400-GT", "Disc_Cup_Masks", seg_fn
                    )
                    fundus_fn = "V{:04d}.jpg".format(index)
                    fundus_path = os.path.join("REFUGE-Validation400", fundus_fn)
                    x, y, label = label_data[fundus_fn]
                    xy = np.array([x, y], dtype=np.float32)
                    fundus = _load_image(fundi.open(fundus_path))
                    image_res = fundus.shape[:2]
                    seg = _load_image(annotations.open(seg_path))
                    seg = _seg_to_label(seg)
                    _transformer = self.builder_config.transformer(image_res)
                    if _transformer is not None:
                        xy = _transformer.transform_point(xy)
                        fundus = _transformer.transform_image(
                            fundus, interp=tf.image.ResizeMethod.BILINEAR
                        )
                        seg = _transformer.transform_image(
                            seg, interp=tf.image.ResizeMethod.NEAREST_NEIGHBOR
                        )
                    yield index, {
                        "fundus": fundus,
                        "segmentation": seg,
                        "label": label,
                        "macular_center": xy,
                        "index": index,
                    }

    def _generate_test_examples(self, fundi):
        def get_seg(image_resolution):
            return np.zeros(image_resolution + (1,), dtype=np.uint8)

        xy = -np.ones((2,), dtype=np.float32)

        with tf.io.gfile.GFile(fundi, "rb") as fundi:
            fundi = zipfile.ZipFile(fundi)
            for index in range(1, 401):
                fundus = _load_image(
                    fundi.open(os.path.join("Test400", "T{:04d}.jpg".format(index)))
                )
                image_res = fundus.shape[:2]
                _transformer = self.builder_config.transformer(image_res)
                if _transformer is not None:
                    fundus = _transformer.transform_image(
                        fundus, interp=tf.image.ResizeMethod.BILINEAR
                    )
                yield index, {
                    "fundus": fundus,
                    "segmentation": get_seg(fundus.shape[:2]),
                    "label": False,
                    "macular_center": xy,
                    "index": index,
                }


__all__ = ["Refuge"]
