# Native Installation

To install the project locally on your system, follow the instructions in this document.

## Project Installation

This section downloads the project, sets up your environment, and installs python dependencies.

### Download project

* **%** `git clone git@github.com:SharedControlAutonomousDriving/scad_tot.git`
* **%** `cd scad_tot`

### Create a virtual environment (recommended)

Using a virtual environment is recommended. The commands below will create and setup the venv for you inside the project's root folder.

* **%** `python3 -m venv venv`
* **%** `source venv/bin/activate`
* **%** `pip install -r requirements.txt`
* **%** `./scripts/setup_venv.sh`
* **%** `deactivate`
* **%** `source venv/bin/activate`

## Marabou Installation

To be able to run the verification code, you'll also need to install the [Marabou Neural Network Verification Framework](https://github.com/NeuralNetworkVerification/Marabou.git). You can skip these steps if you already have Marabou installed on your system.

### Marabou Prerequisites

To compile marabou, `cmake` **>= 3.12** is required. Please see the appropriate section below to install the Marabou's prerequisites on OS X or Linux.

#### OS X (using Homebrew)

Run the following commands to install Marabou's prereqs on OS X. According to Marabou's documentation, `boost` should be installed automatically, but I ran into issues so I installed it manually.

* **%** `brew install cmake`
* **%** `brew install boost`

#### On Linux

On Linux you just need to install `cmake`. Its hard to provide one solution for all Linux distros & package managers, however one good solution is to install it via pip as shown below. If that doesn't work for you, check out [cmake's install page](https://cmake.org/install/) to compile it from source.

* **%** `sudo pip install cmake --upgrade`

### Marabou Install

* Download: **%** `git clone https://github.com/NeuralNetworkVerification/Marabou.git`
* Setup marabou build folder: **%** `mv Marabou .marabou && mkdir .marabou/build && cd .marabou/build`
* Configure: **%** `cmake .. -DBUILD_PYTHON=ON`
  * **IF** you hit an error saying: ['Imported target "openblas" includes non-existent path'](https://github.com/NeuralNetworkVerification/Marabou/issues/380), here is a workaround:
    * run `rm -r ../tools/OpenBLAS-0.3.9`
    * then re-run `cmake .. -DBUILD_PYTHON=ON`
  * NOTE: You may see a ton of warnings. The warnings are OK. Errors are not :)
* Build: **%** `cmake --build .`
* Go back to project root: **%** `cd ../../`

### Add Marabou to PYTHONPATH & JUPYTER_PATH

***NOTE:*** *If you're using a virtual environment (venv) and already ran `./scripts/setup_venv.sh`, then you can skip this section because the paths will be automatically set when you activate the venv and restored when you deactivate.*

Otherwise, the `PYTHONPATH` and `JUPYTER_PATH` environment variables need to be updated to contain the path to Marabou. Replace <PROJECT_ROOT> with the path to this project's root folder.

```zsh
export MARABOU_PATH=<PROJECT_ROOT>/.marabou
export PYTHONPATH=$PYTHONPATH:$MARABOU_PATH
export JUPYTER_PATH=$JUPYTER_PATH:$MARABOU_PATH
```
