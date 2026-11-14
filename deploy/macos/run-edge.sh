#!/bin/zsh
# Starts the Edge API with secrets kept outside the repository.
# Install: copy to ~/.securiot/run-edge.sh and chmod 700.
set -e
. "$HOME/.securiot/edge.env"
cd "${SECURIOT_EDGE_DIR:?set SECURIOT_EDGE_DIR in ~/.securiot/edge.env}"
exec .venv/bin/python -m flask --app wsgi run --host=0.0.0.0 --port=5001
