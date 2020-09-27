# scad_tot Docker Image

The [scad_tot docker image](https://hub.docker.com/r/grese/scad_tot) is an image based on the [grese/marabou](https://hub.docker.com/r/grese/marabou) docker image, also including the [scad_tot](https://github.com/grese/scad_tot) network and verification code.

## Features

Includes all features provided by the [grese/marabou](https://hub.docker.com/r/grese/marabou) image, plus...

* Includes [scad_tot](https://github.com/grese/scad_tot) code

## Usage Examples

The examples below show some different ways to run the image. Feel free to mix and match the different CLI flags as needed.

### Start the Jupyter Lab

You can use the Jupyter Lab web app as shown below.

`docker run -p 9999:9999 grese/marabou`

### Start the Jupyter Lab with a persistent local folder

You can also use the Jupyter Lab web app with a mounted persistent folder from your local machine in the container. This one is arguably the best for local development if you want to save your work.

`docker run -p 9999:9999 -v "$PWD":/home/marabou/work grese/marabou`

### Run as a daemon

You can use the `-d` flag to keep the container running in the background as a daemon.

`docker run -d -p 9999:9999 grese/marabou`

### Run image from command line

Another option is to use the `-i` option to interact with the image from the command line in "interactive mode".

`docker run -i -p 9999:9999 /bin/zsh`

### Other options

Since this image is build on [grese/marabou](https://github.com/grese/marabou-docker), you can also use the additional command line options listed there. It has additional examples for things like providing a persistent SSL certificate, setting a password for the jupyter lab, etc.
