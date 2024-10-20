#!/bin/bash
gcsfuse --implicit-dirs image_generator_request /pascalem/_pyt/CatVTON/data

cd /pascalem

source _pyt/bin/activate

# Check if the environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not activated."
    exit 1
fi

cd _pyt/CatVTON

python3 FileAppRun.py --run_id Parse_agnostic --bucket image_generator_request --config _config3 --token confident-coder-285618-1a0704c69644.json
