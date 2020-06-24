ml_prepare
==========
![Python version range](https://img.shields.io/badge/python-2.7%E2%80%933.6+-blue.svg)
[![License](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Prepares your datasets for ingestion into ML pipelines.

## Install dependencies

    pip install -r requirements.txt

## Install package

    pip install .

## Usage

    $ python -m ml_prepare -h

    usage: python -m ml_prepare [-h] [--version] --dataset
                                {dr_spoc,bmes,dr_spoc_grad_and_no_grad,dr_spoc_no_no_grad}
                                [--retrieve RETRIEVE] [--generate GENERATE]
                                [--tfds TFDS] [--image-height IMAGE_HEIGHT]
                                [--image-width IMAGE_WIDTH]
                                [--image-channels {grayscale,1,rgb,3}]
    
    Prepares your datasets for ingestion into ML pipelines.
    
    optional arguments:
      -h, --help            show this help message and exit
      --version             show program's version number and exit
      --dataset {dr_spoc,bmes,dr_spoc_grad_and_no_grad,dr_spoc_no_no_grad}
      --retrieve RETRIEVE   Retrieve from this directory (or bucket).
      --generate GENERATE   Generate dataset here. Shuffle and then symbolically
                            link to this directory.
      --tfds TFDS           Construct TFrecords and other metadata needed for
                            TensorFlow Datasets, to this directory.
      --image-height IMAGE_HEIGHT
                            Image height
      --image-width IMAGE_WIDTH
                            Image width
      --image-channels {grayscale,1,rgb,3}
                            3 or 'rgb' for red|green|blue (RGB); 1 or 'grayscale'
                            for grayscale

### Matching what tfds automatically does with mnist and friends

    $ export tfds="$HOME"'/tensorflow_datasets/downloads'
    $ export generate="$HOME"'/tensorflow_datasets'

    $ python -m ml_prepare \
             --dataset 'bmes' \
             --retrieve "$HOME"'/BMES123' \
             --tfds "$tfds" \
             --generate "$generate"
    
    $ python -m ml_prepare \
             --dataset 'dr_spoc' \
             --retrieve "$HOME"'/Fundus Photographs for AI' \
             --tfds "$tfds" \
             --generate "$generate"

    $ python -m ml_prepare \
             --dataset 'refuge' \
             --tfds "$tfds" \
             --generate "$generate"

    $ tree -dQ --charset ascii
    .
    |-- "bmes"
    |   `-- "2.0.0.incomplete3PUC97"
    |-- "downloads"
    |-- "dr_spoc"
    |   `-- "2.0.0"
    |-- "dr_spoc_grad_and_no_grad"
    |   `-- "2.0.0"
    |-- "dr_spoc_no_no_grad"
    |   `-- "2.0.0"
    |-- "refuge"
    |   `-- "r224-224-rgb"
    |       `-- "0.0.1"
    `-- "symlinked_datasets"
        |-- "bmes"
        |   |-- "test"
        |   |   |-- "glaucoma"
        |   |   `-- "no_glaucoma"
        |   |-- "train"
        |   |   |-- "glaucoma"
        |   |   `-- "no_glaucoma"
        |   `-- "valid"
        |       |-- "glaucoma"
        |       `-- "no_glaucoma"
        |-- "dr_spoc"
        |   |-- "test"
        |   |   |-- "No gradable image"
        |   |   |-- "non-referable"
        |   |   `-- "referable"
        |   |-- "train"
        |   |   |-- "No gradable image"
        |   |   |-- "non-referable"
        |   |   `-- "referable"
        |   `-- "valid"
        |       |-- "No gradable image"
        |       |-- "non-referable"
        |       `-- "referable"
        |-- "dr_spoc_grad_and_no_grad"
        |   |-- "test"
        |   |   |-- "No gradable image"
        |   |   `-- "gradable"
        |   |-- "train"
        |   |   |-- "No gradable image"
        |   |   `-- "gradable"
        |   `-- "valid"
        |       |-- "No gradable image"
        |       `-- "gradable"
        `-- "dr_spoc_no_no_grad"
            |-- "test"
            |   |-- "non-referable"
            |   `-- "referable"
            |-- "train"
            |   |-- "non-referable"
            |   `-- "referable"
            `-- "valid"
                |-- "non-referable"
                `-- "referable"
    
    56 directories


---

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
