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

To build marabou, `cmake` and `boost` are required. According to Marabou's documentation, it is supposed to install `boost` automatically during the build, but it ran into issues on my machine, so I installed `boost` manually. These commands use `Homebrew`, but you can install them however you like.

* **%** `brew install cmake`
* **%** `brew install boost`

### Marabou Install

* Download: **%** `git clone https://github.com/NeuralNetworkVerification/Marabou.git`
* Setup marabou build folder: **%** `mv Marabou .marabou && mkdir .marabou/build && cd .marabou/build`
* Configure: **%** `cmake .. -DBUILD_PYTHON=ON`
  * If you hit an error saying: ['Imported target "openblas" includes non-existent path'](https://github.com/NeuralNetworkVerification/Marabou/issues/380), run `rm -r ../tools/OpenBLAS-0.3.9`, and then re-run `cmake .. -DBUILD_PYTHON=ON`.
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
