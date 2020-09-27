#!/bin/zsh

# Run the docker image locally
docker run -p 9999:9999 -e SERVER_IP=127.0.0.1 -v "$PWD":/home/marabou/work grese/scad_tot
