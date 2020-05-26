# Marabou Docker Image

The [marabou docker image](https://hub.docker.com/r/grese/marabou) is an ubuntu-based docker image with the [Marabou Neural Network Verification Framework](https://github.com/NeuralNetworkVerification/Marabou.git) pre-installed. It also contains a few convenience features and dependencies useful for neural network verification.

## Features

* Runs Ubuntu 18.04 (bionic)
* Based on [https://hub.docker.com/r/jupyter/tensorflow-notebook](jupyter/tensorflow-notebook)
* [Jupyter Lab](https://jupyter.org), Python3, Numpy, Tensorflow, Pandas, and other useful pip packages
* Marabou & Marabou Python APIs installed
* Standord's [NNet format tools](https://github.com/sisl/NNet) installed
* ZShell is default shell
* A few other helpers and convenience features.

## Usage

### Prerequisites

You'll need [docker](https://www.docker.com/products/docker-desktop) installed, so just install that first if you don't have it.

### Get the image

**%** `docker pull grese/marabou`

### Ways to Run Image

The image supports all of the normal docker command line options. Below are a few examples of ways to run the image. Choose the best one for your use-case.

#### Run the JupyterLab browser app

**%** `docker run -p 9999:9999 grese/marabou`

Once it is running, visit https://localhost:9999?token=TOKEN in your browser (TOKEN will be printed in the console when server starts)

##### Run with a shared folder from your local machine (-v option)

**%** `docker run -p 9999:9999 -v "$PWD":/home/marabou/work grese/marabou`

#### Run container command line

**%** `docker run -p 9999:9999 -it grese/marabou /bin/zsh`

#### Run as daemon

**%** `docker run -d -p 9999:9999 grese/marabou`

### Using Marabou

Marabou has been pre-installed in `~/.bin/marabou` along with all of the source code. Below are a couple of examples of how Marabou can be used within the container.

#### Run Marabou Binary

A ZSH command has been added for Marabou, so you can just run `marabou` from the command line. See examples below.

**%** `marabou --help`

**%** `marabou --input=NNET_FILE --property=PROPERTY_FILE`

### Using Marabou's Python APIs

You can just directly import and use the Marabou Python APIs as you would any other Python module. Below is a quick example using Python's interactive shell.

**%** `python`

```python
> from maraboupy import Marabou
> nnet = Marabou.read_nnet('path/to/nnet_file')
```
