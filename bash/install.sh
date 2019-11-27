#!/bin/bash

# Download project data
echo "Download project data"
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1uQdGE-4oNEncaQ2T4oy1REiyB5IWS0as' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1uQdGE-4oNEncaQ2T4oy1REiyB5IWS0as" -O data.zip && rm -rf /tmp/cookies.txt
unzip data.zip
rm -r data.zip

# Setup environment
echo "Prepare venv"
module load python3/intel/3.5.3
cd ..
python3 -m virtualenv mvenv
source ./mvenv/bin/activate

# Install requirements
echo "Install requirements"
cd dl-master-voices
pip install -r requirements.txt

export PYTHONPATH="$(pwd)"


