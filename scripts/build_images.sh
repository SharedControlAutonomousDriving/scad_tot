#!/bin/zsh

# Builds docker images for the project
#
# Usage: ./scripts/build_images.sh [options]
#
# Options: 
#   - p: push images after build

# image paths and names
MARABOU_DOCKER_PATH=./docker/marabou
MARABOU_DOCKER_IMAGE=grese/marabou
SCAD_TOT_DOCKER_PATH=./docker/scad_tot
SCAD_TOT_DOCKER_IMAGE=grese/scad_tot

# push images flag
PUSH_IMAGES=0

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

# Build docker images
build_image() {
	[ "$3" = "0" ] && local CACHE_ARG="--no-cache" || local CACHE_ARG=""
	cd $1
	docker build $CACHE_ARG -t $2 .
	cd - > /dev/null
}

echo "Building docker images..."
build_image $MARABOU_DOCKER_PATH $MARABOU_DOCKER_IMAGE
build_image $SCAD_TOT_DOCKER_PATH $SCAD_TOT_DOCKER_IMAGE 0

# Push images to docker hub (requires -p flag)
push_image() { docker push $1 }

if [ "$PUSH_IMAGES" -eq "1" ]; then
    echo "Pushing images to docker hub..."
    push_image $MARABOU_DOCKER_IMAGE
    push_image $SCAD_TOT_DOCKER_IMAGE
fi
