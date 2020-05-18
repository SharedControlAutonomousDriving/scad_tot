# Based on grese/marabou image
FROM grese/marabou:latest

# Docker arguments
ARG MARABOU_USER=marabou
ARG MARABOU_HOME=/home/marabou
ARG SCAD_TOT_URL=https://github.com/grese/scad_tot.git
ARG SCAD_TOT_PATH=${MARABOU_HOME}/scad_tot

# Environment variables
ENV SCAD_TOT_PATH=${SCAD_TOT_PATH}

# Download and install scad_tot project
RUN cd ${MARABOU_HOME} && git clone ${SCAD_TOT_URL} && chown -R ${MARABOU_USER}:${NB_GROUP} ${SCAD_TOT_PATH}
