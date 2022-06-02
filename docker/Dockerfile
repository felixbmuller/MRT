FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

RUN apt-get update && apt-get install -y apt-file && apt upgrade -y
RUN apt install -y build-essential curl git cmake pkg-config checkinstall

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh
RUN bash Miniconda3-4.7.10-Linux-x86_64.sh -b -p /opt/miniconda3
#RUN conda update -n base -c defaults conda
RUN conda install -y pip numpy

RUN conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit==9.0 -c pytorch
RUN pip install torch-dct
RUN pip install transforms3d matplotlib pygame pyopengl
# install fmpeg does not work without update
RUN apt-get update
RUN apt-get install -y ffmpeg
RUN conda install numba

# dataset is included via path.json, link it to the location the scripts expect it
# did this manually, how to do it in Dockerfile?
#RUN ln -s /home/user/all_asfamc /home/user/mrt/mocap/