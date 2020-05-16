# Marabou Docker Image

The [marabou docker image](https://hub.docker.com/r/grese/marabou) is an ubuntu-based docker image with the [Marabou Neural Network Verification Framework](https://github.com/NeuralNetworkVerification/Marabou.git) pre-installed. It also contains a few convenience features and dependencies useful for neural network verification.

## Features

* Based on Ubuntu 20.04 (focal)
* Python3 installed
* Numpy, Tensorflow, Pandas, and other useful pip packages
* Marabou & Marabou Python APIs installed
* Standord's [NNet format tools](https://github.com/sisl/NNet) installed
* ZShell is default shell
* Alias to run marabou as a command

## Usage

### Get the image

**%** `docker pull grese/marabou`

### Run the image

**%** `docker run -it grese/marabou /bin/zsh`

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
