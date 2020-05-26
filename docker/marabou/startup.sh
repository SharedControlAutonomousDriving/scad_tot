#!/bin/zsh

STARTUP_ARGS="--NotebookApp.port=9999 --NotebookApp.notebook_dir=$MARABOU_HOME --ContentsManager.root_dir=$MARABOU_HOME --FileContentsManager.root_dir=$MARABOU_HOME"

if [ ! -z "$SERVER_IP" ]; then
    STARTUP_ARGS="$STARTUP_ARGS --NotebookApp.ip=$SERVER_IP"
fi

if [ ! -z "$SERVER_TOKEN" ]; then
    STARTUP_ARGS="$STARTUP_ARGS --NotebookApp.password=$SERVER_TOKEN"
fi

if [ ! -z "$SERVER_CERTFILE" ] && [ ! -z "$SERVER_KEYFILE" ]; then
    STARTUP_ARGS="$STARTUP_ARGS --NotebookApp.certfile=$SERVER_CERTFILE --NotebookApp.keyfile=$SERVER_KEYFILE"
    GEN_CERT=no
else
    GEN_CERT=yes
fi

if [ "$SERVER_MODE" = "lab" ]; then
    STARTUP_ARGS="jupyter lab $STARTUP_ARGS"
else
    STARTUP_ARGS="jupyter notebook $STARTUP_ARGS"
fi

eval "/usr/local/bin/start.sh $STARTUP_ARGS"
