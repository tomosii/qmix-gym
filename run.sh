set -x 
IMAGE_NAME_TAG=qmix-gym
#docker run  --rm --net=host -it --gpus '"device=0"' $IMAGE_NAME_TAG bash
sudo docker run -u `id -u`:`id -g` --rm --net=host -it --gpus all  -v /tmp:/host/tmp -v /etc/group:/etc/group:ro -v /etc/passwd:/etc/passwd:ro -v `pwd`:/work $IMAGE_NAME_TAG bash
