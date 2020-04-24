# Native Installation

To install the project locally on your system (instead of using the Docker image), follow the instructions in this document.

## Project Installation

This section downloads the project, sets up your environment, and installs python dependencies.

First, clone the repo and move into the cloned directory.

* `git clone https://github.com/grese/scad_tot.git`
* `cd scad_tot`

*Recommended:&nbsp;* Create a virtual environment and install the python dependencies.

* `python3 -m venv venv`
* `source venv/bin/activate`
* `pip install -r requirements.txt`

## Marabou Installation

To be able to run the verification code, you'll also need to install the [Marabou Neural Network Verification Framework](https://github.com/NeuralNetworkVerification/Marabou.git). You can skip these steps if you already have Marabou installed on your system.

### Marabou Prerequisites

To build marabou, `cmake` and `boost` are required. According to Marabou's documentation, it is supposed to install `boost` automatically during the build, but it ran into issues on my machine, so I installed `boost` manually. These commands use `Homebrew`, but you can install them however you like.

* `brew install cmake`
* `brew install boost`

### Marabou Install

* Download: `wget https://github.com/NeuralNetworkVerification/Marabou/archive/master.zip -O marabou.zip`
* Decompress: `unzip marabou.zip && mv Marabou-master .marabou && rm marabou.zip`
* Create build folder: `cd .marabou && mkdir build && cd build`
* Configure: `cmake .. -DCMAKE_BUILD_TYPE=Release`
* Build: `cmake --build .`
* Go back to project root: `cd ../../`

### Add Marabou to PYTHONPATH & JUPYTER_PATH

We need to tell Python where to find Marabou by updating the `PYTHONPATH` and `JUPYTER_PATH` environment variables. Since we are in a virtual environment, the commands can be added to the venv's `activate` script.

Open `./venv/bin/activate` in your favorite text editor, and add the following lines (replacing <PROJECT_ROOT> with the absolute path to the project's root folder):

```zsh
export MARABOU_PATH=<PROJECT_ROOT>/.marabou
export PYTHONPATH=$PYTHONPATH:$MARABOU_PATH
export JUPYTER_PATH=$JUPYTER_PATH:$MARABOU_PATH
```

After adding these to the venv's activate script, run the following commands:

* Exit the venv: `deactivate`
* Enter the venv: `source venv/bin/activate`
