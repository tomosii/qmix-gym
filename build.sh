#!/bin/bash

# sudo bash build.sh

IMAGE_NAME_TAG=qmix-gym
DOCKER_FILE=Dockerfile
docker build -t ${IMAGE_NAME_TAG} -f ${DOCKER_FILE} .
