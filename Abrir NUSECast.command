#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "Falta instalar el entorno. Consulta LEEME.md."
  exit 1
fi
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
