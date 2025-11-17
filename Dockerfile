FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

WORKDIR /workspaces
RUN apt-get update && \
    apt-get install -y \
    software-properties-common \
    nano \
    vim \
    git

# Python installation
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt-get install -y python3.12 python3-pip 

RUN pip install poetry --break-system-packages


RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc && \
    sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' ~/.bashrc || true
# add-apt-repository ppa:deadsnakes/ppa && \
# apt-get update && \
# apt-get install -y python3.12 python3.12-distutils


# RUN wget https://bootstrap.pypa.io/get-pip.py && python3.12 get-pip.py