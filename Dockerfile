FROM nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

RUN apt-key del 7fa2af80 && \
     apt-key del F60F4B3D7FA2AF80 && \
     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && \
     apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN apt-get install -y libjpeg-dev zlib1g-dev
RUN pip3 install torch torchvision torchaudio
RUN pip3 install gym
RUN pip3 install wandb

WORKDIR /work

#ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
