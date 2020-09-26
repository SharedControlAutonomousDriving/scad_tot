# scad_tot ![Build & Push Docker Image](https://github.com/grese/scad_tot/workflows/Build%20&%20Push%20Docker%20Image/badge.svg)

[Safe-SCAD](https://www.york.ac.uk/assuring-autonomy/projects/safe-scad/) "TakeOverTime" neural network &amp; verification.

## Project Structure

* **Network:** The neural network related code can be found in the [network folder](./network).
* **Verification:** The network verification code can be found in the [verification folder](./verification).
* **Data:** Data can be found in the [data folder](./data).

## Installation

You have two options here. *Option 1* is to install the project natively, which is recommended for most developers. *Option 2* is to use the docker image, which is mostly intended for running the project in the cloud, however could potentially be used for local development. Installing locally is probably the most comfortable development experience for most developers that aren't familiar with Docker because containers are intended to be ephemeral. Choose whichever option you prefer though :).

### Option 1: Native Install

This option installs the project natively on your system. Using a `venv` is recommended, but not required. Detailed instructions for this option can be found here: [Native Installation Instructions](./docs/NATIVE_INSTALL.md).

### Option 2: Docker Image

This option runs the project from within a docker container. It is the quickest way to get started. Instructions for this method can be found here: [Docker Installation Instructions](./docs/DOCKER_INSTALL.md)

## Download Dataset

Run the following command to download the dataset.

**%** `./scripts/download_data.sh`

## Running Verification

### Robustness

* To run in a jupyter notebook, see `./verification/robustness.ipynb`
* To run from command line, see `./scripts/robustness.sh` and `./scripts/robustness_asym.sh`
  * View CLI options with `./verification/robustness.py --help`

### Sensitivity

* To run in a jupyter notebook, see `./verification/sensitivity.ipynb`
* To run from command line, see `./scripts/sensitivity.sh` and `./scripts/sensitivity_asym.sh`
  * View CLI options with `./verification/sensitivity.py --help`

### Generating & Verifying Regions

* To run in a jupyter notebook, see `./verification/clustering.ipynb`
