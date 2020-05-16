# Based on ubuntu 18.04
FROM ubuntu:bionic

# Dockerfile arguments
ARG USER=ubuntu
ARG HOME=/home/${USER}
ARG MARABOU_PATH=${HOME}/.bin/marabou
ARG MARABOU_URL=https://github.com/NeuralNetworkVerification/Marabou/archive/master.zip
ARG NNET_TOOLS_PATH=${HOME}/.bin/nnet_tools
ARG NNET_TOOLS_URL=https://github.com/sisl/NNet/archive/master.zip

# Environment variables
ENV HOME=${HOME}
ENV USER=${USER}
ENV MARABOU_PATH=${MARABOU_PATH}
ENV NNET_TOOLS_PATH=${NNET_TOOLS_PATH}
ENV PYTHONPATH="${PYTHONPATH}:${MARABOU_PATH}:${NNET_TOOLS_PATH}"
ENV JUPYTER_PATH="${JUPYTER_PATH}:${MARABOU_PATH}:${NNET_TOOLS_PATH}"
ENV TERM="xterm"

# Install apt packages
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y && apt-get autoclean && apt-get autoremove -y
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y sudo apt-utils && \
    apt-get install -y build-essential libssl-dev protobuf-compiler libprotoc-dev && \
    apt-get install -y zsh vim git zip wget cmake && apt-get install -y python3-dev python3-pip python3-venv

# Create and configure user account
RUN useradd -m ${USER} -s /bin/zsh && adduser ${USER} sudo && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN mkdir -p ${HOME}/.bin && wget -q -O ${HOME}/.bin/git-prompt.sh https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh
RUN chown -R ${USER}:${USER} ${HOME}/.bin
COPY --chown=${USER}:${USER} ./zshrc ${HOME}/.zshrc

# fix sudo error "setrlimit(RLIMIT_CORE): Operation not permitted"
RUN touch /etc/sudo.conf && echo "Set disable_coredump false" >> /etc/sudo.conf

# Setup python and pip
RUN ln -s /usr/bin/python3 /usr/local/bin/python
RUN ln -s /usr/bin/pip3 /usr/local/bin/pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install python packages
RUN pip3 install --no-cache-dir pybind11 pytest numpy tensorflow pandas
RUN pip3 install --no-cache-dir onnx onnxruntime

# Download, unzip, build, and setup marabou
RUN cd /tmp && wget -q -O marabou.zip ${MARABOU_URL} &&  \
    unzip -q marabou.zip && mv Marabou-master $MARABOU_PATH && rm -rf marabou.zip
RUN cd $MARABOU_PATH && mkdir build
RUN cd $MARABOU_PATH/build && cmake .. -DBUILD_PYTHON=ON && cmake --build .
RUN chown -R $USER:$USER $MARABOU_PATH

# Download, unzip, and setup nnet_tools
RUN cd /tmp && wget -q -O nnet_tools.zip ${NNET_TOOLS_URL} &&  \
    unzip -q nnet_tools.zip && mv NNet-master $NNET_TOOLS_PATH && rm -rf nnet_tools.zip
RUN chown -R $USER:$USER $NNET_TOOLS_PATH

# Configure default user, cwd, and shell for docker
USER ${USER}
WORKDIR ${HOME}
CMD /bin/zsh
