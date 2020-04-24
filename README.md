# scad_tot

[Safe-SCAD](https://www.york.ac.uk/assuring-autonomy/projects/safe-scad/) "TakeOverTime" neural network &amp; verification.

## Project Structure

* **Network:** The neural network related code can be found in the [network folder](./network).
* **Verification:** The network verification code can be found in the [verification folder](./verification).
* **Data:** Data can be found in the [data folder](./data).

## Installation

You have two options here. Option 1 is to use the docker image. Option 2 is to install the project natively. Choose whichever you prefer.

### Option 1: Docker Image

This option runs the project from within a docker container. It is the quickest way to get started. Instructions for this method can be found here: [Docker Installation Instructions](./docs/DOCKER_INSTALL.md)

### Option 2: Native Install

This option installs the project natively on your system. Using a `venv` is recommended, but not required. Detailed instructions for this option can be found here: [Native Installation Instructions](./docs/NATIVE_INSTALL.md).

## Download Dataset

This repository does not include the dataset because it is too large. Download the CSV file and place it in `./data/rt_all_data.csv`.
