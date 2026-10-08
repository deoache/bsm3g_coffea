#!/bin/bash

LCG_SETUP="/cvmfs/sft.cern.ch/lcg/views/LCG_110/x86_64-el9-gcc13-opt/setup.sh"
RUCIO_DIR="$HOME/.local/rucio-lcg110"

source "$LCG_SETUP"

if ! PYTHONPATH="$RUCIO_DIR:$PYTHONPATH" python -c "import rucio" &>/dev/null; then
    echo "Installing rucio-clients..."
    python -m pip install --target="$RUCIO_DIR" rucio-clients
fi

export PYTHONPATH="$RUCIO_DIR:$PYTHONPATH"