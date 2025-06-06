#!/bin/bash
cd /pascalem

source _pyt/bin/activate

# Check if the environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Virtual environment not activated."
    exit 1
fi

cd _pyt/CatVTON

python3 FileSharing_Run.py --run_id CatVTON --bucket image_generator_request --config _config3 --token confident-coder-285618-1a0704c69644.json 
