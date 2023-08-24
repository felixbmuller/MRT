FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
ENV PATH /opt/miniconda3/bin:$PATH
ENV CPLUS_INCLUDE_PATH /opt/miniconda3/include

#RUN apt install wget
#RUN apt-key del 7fa2af80
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/cuda-keyring_1.0-1_all.deb
#RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update 
RUN apt-get install -y apt-file && apt upgrade -y
RUN apt install -y build-essential curl git cmake pkg-config checkinstall

RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-4.7.10-Linux-x86_64.sh
RUN bash Miniconda3-4.7.10-Linux-x86_64.sh -b -p /opt/miniconda3
#RUN conda update -n base -c defaults conda
RUN conda install -y pip numpy pandas

RUN conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit==9.0 -c pytorch
RUN pip install torch-dct
RUN pip install transforms3d matplotlib pygame pyopengl
RUN pip install open3d==0.15.2
RUN pip install trimesh
# install fmpeg does not work without update
RUN apt-get update
RUN apt-get install -y ffmpeg
#RUN pip install mlflow einops seaborn jupyterlab fire numba tensorboard
#RUN pip install git+https://github.com/jutanke/ndms.git

# dataset is included via path.json, link it to the location the scripts expect it
# did this manually, how to do it in Dockerfile?
#RUN ln -s /home/user/all_asfamc /home/user/mrt/mocap/
#RUN ln -s /home/user/MultiPersonTestSet /home/user/mrt/mupots3d/data