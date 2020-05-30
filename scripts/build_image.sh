#!/bin/zsh

# Builds and optionally pushes docker image for the project
#
# Usage: ./scripts/build_image.sh [options]
#
# Options: 
#   - p: push image after build

# image path and name
SCAD_TOT_DOCKER_PATH=./docker/.
SCAD_TOT_DOCKER_IMAGE=grese/scad_tot

# push image flag
PUSH_IMAGE=0

# usage function
usage() { echo "Usage: $0 [-p]" 1>&2; exit 1; }

# parse CLI options
while getopts "hp" o; do
  case "${o}" in
  p)   PUSH_IMAGES=1;;
  h|*) usage;;
  esac
done
shift $((OPTIND-1))

echo "Building docker image..."
docker build --no-cache $SCAD_TOT_DOCKER_PATH -t $SCAD_TOT_DOCKER_IMAGE

# Push images to docker hub (requires -p flag)
if [ "$PUSH_IMAGES" -eq "1" ]; then
    echo "Pushing images to docker hub..."
    docker push $MARABOU_DOCKER_IMAGE
    docker push $SCAD_TOT_DOCKER_IMAGE
fi
