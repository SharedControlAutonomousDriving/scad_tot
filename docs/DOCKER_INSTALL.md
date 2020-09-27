# Docker Installation

The [docker image](https://hub.docker.com/r/grese/scad_tot) provides everything you need to run the project packed into a docker container.

## Prerequisites

* You'll need [Docker Desktop](https://www.docker.com/products/docker-desktop) installed. So, install that first if you don't have it already.

## Installation

* Pull the image: **%** `docker pull grese/scad_tot`

## Usage

The examples below show some different ways to run the image. Feel free to mix and match the different CLI flags as needed.

### Start the Jupyter Lab

You can use the Jupyter Lab web app as shown below. Visit `http://localhost:9999` in your browser after running it.

`docker run -p 9999:9999 grese/marabou`

### Start the Jupyter Lab with a persistent local folder

You can also use the Jupyter Lab web app with a mounted persistent folder from your local machine. This one is arguably the best for local development if you want to save your work.

`docker run -p 9999:9999 -v "$PWD":/home/marabou/work grese/marabou`

### Run as a daemon

You can use the `-d` flag to keep the image running in the background as a daemon.

`docker run -d -p 9999:9999 grese/marabou`

### Run image from command line

Another option is to use the `-i` option to interact with the image from the command line in "interactive mode".

`docker run -i -p 9999:9999 /bin/zsh`

### Other options

Since this image is build on [grese/marabou](https://github.com/grese/marabou-docker), you can also use the additional command line options listed there.
