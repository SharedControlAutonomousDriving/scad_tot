# Based on grese/marabou image
FROM grese/marabou:latest

# Docker arguments
ARG MARABOU_USER=marabou
ARG MARABOU_HOME=/home/marabou
ARG NB_GROUP=users
ARG SCAD_TOT_URL=https://github.com/grese/scad_tot/archive/master.zip
ARG SCAD_TOT_PATH=${MARABOU_HOME}/scad_tot
ARG SCAD_TOT_IGNORE='docker docs'
ARG SCAD_TOT_DATA_URL=https://drive.google.com/uc?authuser=0&export=download&id=1KkoEaLUsmV916TO9FOAuGmN_uy2o9e-t

# Environment variables
ENV NB_USER=${MARABOU_USER}
ENV SCAD_TOT_PATH=${SCAD_TOT_PATH}

# install scad_tot project
RUN cd /tmp && wget -q -O scad_tot.zip ${SCAD_TOT_URL} && unzip -q scad_tot.zip \
    && mv scad_tot-master ${SCAD_TOT_PATH} && rm -rf scad_tot.zip
RUN cd ${SCAD_TOT_PATH} && rm -rf ${SCAD_TOT_IGNORE}

# download scad data
RUN mkdir -p ${SCAD_TOT_PATH}/data && wget -q -O ${SCAD_TOT_PATH}/data/data.csv ${SCAD_TOT_DATA_URL}

# fix permissions
RUN chown -R ${MARABOU_USER}:${NB_GROUP} ${SCAD_TOT_PATH}
