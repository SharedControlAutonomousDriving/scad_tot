# scad_tot Docker Image

The [scad_tot docker image](https://hub.docker.com/r/grese/scad_tot) is an image based on the [grese/marabou](https://hub.docker.com/r/grese/marabou) docker image, also including the [scad_tot](https://github.com/grese/scad_tot) network and verification code.

## Features

Includes all features provided by the [grese/marabou](https://hub.docker.com/r/grese/marabou) image, plus...

* Includes [scad_tot](https://github.com/grese/scad_tot) code

## Usage

### Get the image

**%** `docker pull grese/scad_tot`

### Start a container

You can start the container using all of the normal docker command line options. Here are a couple of examples:

**%** `docker run -p 9999:9999 grese/scad_tot`

Copy the URL printed in the console, and access https://localhost:9999?token=TOKEN in your browser.

### Start a container with a mounted folder from host machine

**%** `docker run -p 9999:9999 -v /path/to/local/folder:/home/marabou/work grese/scad_tot`

Note: replace "/path/to/local/folder" with the folder you want to mount in the container
