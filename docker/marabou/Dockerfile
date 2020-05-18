# Based on ubuntu 18.04
FROM jupyter/tensorflow-notebook:latest

# Dockerfile arguments
ARG MARABOU_URL=https://github.com/NeuralNetworkVerification/Marabou/archive/master.zip
ARG MARABOU_USER=marabou
ARG MARABOU_HOME=/home/${MARABOU_USER}
ARG MARABOU_PATH=${MARABOU_HOME}/.bin/marabou
ARG NNET_URL=https://github.com/sisl/NNet/archive/master.zip
ARG NNET_PATH=${MARABOU_HOME}/.bin/nnet
ARG OLD_USER=jovyan
ARG NB_GROUP=users

# Configure default user, cwd, and shell for docker
ENV NB_USER=${MARABOU_USER}
ENV GEN_CERT=yes
ENV MARABOU_PATH=${MARABOU_PATH}
ENV NNET_PATH=${NNET_PATH}
ENV PYTHONPATH="$PYTHONPATH:$MARABOU_PATH:$NNET_PATH"
ENV JUPYTER_PATH="$JUPYTER_PATH:$MARABOU_PATH:$NNET_PATH"
ENV TERM="xterm"

# run as root (will change to marabou)
USER root

# Install aptitude dependencies
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y \
    && apt-get install -y build-essential libssl-dev protobuf-compiler libprotoc-dev \
    && apt-get install -y zsh zip wget cmake && apt-get install -y libboost-all-dev python3-dev

# Create and configure user account (TODO: require password for sudo)
RUN mv /home/${OLD_USER} ${MARABOU_HOME}
RUN usermod ${OLD_USER} -l ${MARABOU_USER} -d ${MARABOU_HOME} -s /bin/zsh -g ${NB_GROUP} -G sudo && echo "${MARABOU_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN rm -rf /home/${OLD_USER}
# make marabou directory and copy folders
RUN mkdir -p /home/marabou/.bin
COPY ./git-prompt.sh /home/marabou/.bin/git-prompt.sh
COPY ./zshrc /home/marabou/.zshrc

# Upgrade pip and install packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pybind11 pytest
RUN pip install --no-cache-dir onnx onnxruntime

# Download, unzip, build, and setup marabou
RUN cd /tmp && wget -q -O marabou.zip ${MARABOU_URL} && unzip -q marabou.zip \
    && mv Marabou-master ${MARABOU_PATH} && rm -rf marabou.zip
RUN mkdir ${MARABOU_PATH}/build && cd ${MARABOU_PATH}/build && cmake .. -DBUILD_PYTHON=ON && cmake --build .

# Download, unzip, and setup nnet
RUN cd /tmp && wget -q -O nnet.zip ${NNET_URL} && unzip -q nnet.zip \
    && mv NNet-master ${NNET_PATH} && rm -rf nnet.zip

# fix permissions
RUN chown -R ${MARABOU_USER}:${NB_GROUP} ${MARABOU_HOME}

WORKDIR ${MARABOU_HOME}
CMD /usr/local/bin/start-notebook.sh \ 
    --NotebookApp.port=9999 \ 
    --NotebookApp.notebook_dir=${MARABOU_HOME} \ 
    --ContentsManager.root_dir=${MARABOU_HOME} \ 
    --FileContentsManager.root_dir=${MARABOU_HOME}
