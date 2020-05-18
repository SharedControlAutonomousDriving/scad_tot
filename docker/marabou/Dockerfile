# Based on ubuntu 18.04
FROM jupyter/tensorflow-notebook:latest

# Dockerfile arguments
# ARG MARABOU_USER='marabou'
# ARG MARABOU_UID=${NB_UID}
# ARG MARABOU_GID=${NB_GID}
# ARG NB_USER=${MARABOU_USER}
# ARG HOME=/home/${MARABOU_USER}
ARG NB_HOME=/home/${NB_USER}
ARG MARABOU_URL=https://github.com/NeuralNetworkVerification/Marabou/archive/master.zip
ARG MARABOU_INSTALL_PATH=${NB_HOME}/.bin/marabou
ARG MARABOU_USER=marabou
ARG MARABOU_HOME=/home/${MARABOU_USER}
ARG MARABOU_PATH=${MARABOU_HOME}/.bin/marabou
ARG NNET_URL=https://github.com/sisl/NNet/archive/master.zip
ARG NNET_INSTALL_PATH=${NB_HOME}/.bin/nnet
ARG NNET_PATH=${MARABOU_HOME}/.bin/nnet

# Environment variables
# ENV HOME=${HOME}
# ENV NB_USER=${NB_USER}
# ENV MARABOU_PATH=${MARABOU_PATH}
# ENV NNET_PATH=${NNET_PATH}
# ENV PYTHONPATH="${PYTHONPATH}:${MARABOU_PATH}:${NNET_PATH}"
# ENV JUPYTER_PATH="${JUPYTER_PATH}:${MARABOU_PATH}:${NNET_PATH}"
ENV TERM="xterm"

# temporarily switch to root user
USER root

# Install aptitude dependencies
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y \
    && apt-get install -y build-essential libssl-dev protobuf-compiler libprotoc-dev \
    && apt-get install -y zsh zip wget cmake && apt-get install -y libboost-all-dev python3-dev

# Create and configure user account (TODO: require password for sudo)
# RUN useradd -m ${USER} -s /bin/zsh && adduser ${USER} sudo && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# RUN chsh -s /bin/zsh ${USER} && echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# RUN useradd -m ${MARABOU_USER} -u ${MARABOU_UID} -g ${MARABOU_GID} -G sudo -s /bin/zsh && echo "${MARABOU_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# RUN usermod jovyan -l ${MARABOU_USER} -d ${HOME} -m -s /bin/zsh -G sudo && echo "${MARABOU_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# USER ${MARABOU_USER}
# RUN mkdir -p ${HOME}/.bin ${HOME}/work && wget -q -O ${HOME}/.bin/git-prompt.sh https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh
RUN mkdir -p ${NB_HOME}/.bin && wget -q -O ${NB_HOME}/.bin/git-prompt.sh https://raw.githubusercontent.com/git/git/master/contrib/completion/git-prompt.sh
# RUN chown -R ${MARABOU_USER}:${MARABOU_GID} ${HOME}/.bin
# COPY --chown=${MARABOU_USER}:${MARABOU_GID} ./zshrc ${HOME}/.zshrc
RUN chown -R ${NB_USER}:${NB_GID} ${HOME}/.bin
COPY --chown=${NB_USER}:${NB_GID} ./zshrc ${HOME}/.zshrc

# Upgrade pip and install packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pybind11 pytest
RUN pip install --no-cache-dir onnx onnxruntime

# Download, unzip, build, and setup marabou
RUN cd /tmp && wget -q -O marabou.zip ${MARABOU_URL} && unzip -q marabou.zip \
    && mv Marabou-master ${MARABOU_INSTALL_PATH} && rm -rf marabou.zip
RUN cd ${MARABOU_INSTALL_PATH} && mkdir build
RUN cd ${MARABOU_INSTALL_PATH}/build && cmake .. -DBUILD_PYTHON=ON && cmake --build .
RUN chown -R ${NB_USER}:${NB_GID} ${MARABOU_INSTALL_PATH}

# Download, unzip, and setup nnet
RUN cd /tmp && wget -q -O nnet.zip ${NNET_URL} && unzip -q nnet.zip \
    && mv NNet-master ${NNET_INSTALL_PATH} && rm -rf nnet.zip
# RUN chown -R ${MARABOU_USER}:${MARABOU_GID} $NNET_PATH
RUN chown -R ${NB_USER}:${NB_GID} ${NNET_INSTALL_PATH}

# Configure default user, cwd, and shell for docker
WORKDIR ${MARABOU_USER}
# CMD start-notebook.sh -e NB_USER=marabou -e CHOWN_HOME_OPTS='-R' -e GEN_CERT=yes
ENV NB_USER=${MARABOU_USER}
ENV CHOWN_HOME_OPTS='-R'
# ENV GEN_CERT=yes
ENV MARABOU_PATH=${MARABOU_PATH}
ENV NNET_PATH=${NNET_PATH}
ENV PYTHONPATH="$PYTHONPATH:$MARABOU_PATH:$NNET_PATH"
ENV JUPYTER_PATH="$JUPYTER_PATH:$MARABOU_PATH:$NNET_PATH"

CMD /usr/local/bin/start-notebook.sh --NotebookApp.port=9999
