# $ git clone https://github.com/jeongho9413/lidargsu.git && cd lidargsu
# $ docker build --tag lidargsu_test .
# $ docker run -it --gpus all -v /home/jeongho9413/Downloads/test_docker_20250616:/workspace --name lidargsu_test_container lidargsu_test
# $ docker start lidargsu_test_container
# $ docker attach lidargsu_test_container


# base image with CUDA and cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# install Python and pip
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    build-essential curl fzf htop sudo tmux wget zsh \
    ninja-build libsparsehash-dev ffmpeg \
    git ca-certificates
RUN apt-get install -y python3 python3-pip && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

RUN apt-get autoremove -y && \
    apt-get clean -y  && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# set default shell
CMD ["/bin/bash"]
