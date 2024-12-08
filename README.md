
# HiFi-GAN Implementation

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains my implementation of HiFi-GAN

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw3_nv).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

If you want to fine-tune model with augmentations (like our best model), then you need to add `from_pretrained` parameter to trainer section in train config and specify in it the path to your checkpoint. Also you need to change config in `transforms` section to `example_only_instance_augs`.

To download checkpoint run

```bash
python download_checkpoint.py
```

To run inference (save predictions) on either on audio dataset or on text dataset:

```bash
python synthesize.py inferencer.save_path='<path to saving directory>' inferencer.data_dir_path='<path to data directory>' 
```

You can also run inference on a custom text:
```bash
python synthesize.py inferencer.text_from_cli='I love DLA' inferencer.save_path='<path to saving directory>'
```
The resulting `.wav` will be saved in a file named `cli_input_gen.wav`
## How To Measure Metrics

The WV-MOS will be printed out when using `synthesize.py`

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)