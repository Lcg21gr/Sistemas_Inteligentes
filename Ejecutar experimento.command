#!/bin/zsh
set -eu
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  print "Falta instalar el entorno. Consulta LEEME.md."
  exit 1
fi
.venv/bin/python -m nusecast.cli run --profile demo
print "Experimento terminado. Los resultados están en outputs."
