#!/bin/bash

JUPYTER_PORT=28293

echo "[InternetShortcut]" > jupyter.url
echo "URL=http://$HOSTNAME:$JUPYTER_PORT" >> jupyter.url
echo "NAME=Jupyter Lab" >> jupyter.url

jupyter lab --no-browser --port $JUPYTER_PORT
