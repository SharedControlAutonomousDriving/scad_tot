# Based on grese/marabou image
FROM grese/marabou:latest

# Docker arguments
ARG MARABOU_USER=marabou
ARG MARABOU_HOME=/home/marabou
ARG NB_GROUP=users
ARG SCAD_TOT_URL=https://github.com/SharedControlAutonomousDriving/scad_tot/archive/master.zip
ARG SCAD_TOT_PATH=${MARABOU_HOME}/scad_tot
ARG SCAD_TOT_IGNORE='docker docs'

# Environment variables
ENV NB_USER=${MARABOU_USER}
ENV SCAD_TOT_PATH=${SCAD_TOT_PATH}

# install scad_tot project
RUN cd /tmp && wget -q -O scad_tot.zip ${SCAD_TOT_URL} && unzip -q scad_tot.zip \
    && mv scad_tot-master ${SCAD_TOT_PATH} && rm -rf scad_tot.zip
RUN cd ${SCAD_TOT_PATH} && rm -rf ${SCAD_TOT_IGNORE}

# fix permissions
RUN chown -R ${MARABOU_USER}:${NB_GROUP} ${SCAD_TOT_PATH}
