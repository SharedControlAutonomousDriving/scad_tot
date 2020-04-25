# Based on grese/marabou image
FROM grese/marabou:latest

# Docker arguments
ARG SCAD_TOT_URL=https://github.com/grese/scad_tot.git
ARG SCAD_TOT_PATH=${HOME}/scad_tot

# Environment variables
ENV SCAD_TOT_PATH=${SCAD_TOT_PATH}

# Download and install scad_tot project
RUN cd ${HOME} && git clone ${SCAD_TOT_URL} && chown -R ${USER}:${USER} ${SCAD_TOT_PATH}
