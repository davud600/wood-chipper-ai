#!/bin/bash

set -e

# Constants
VENV_NAME=".venv"
ZIP_NAME="./data/dataset.tar.gz"
DATA_DIR="./data/dataset"
REQUIREMENTS="requirements.txt"

echo "🟢 Setting up virtual environment..."
python -m venv $VENV_NAME
source $VENV_NAME/bin/activate

echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r $REQUIREMENTS

echo "📁 Unzipping dataset..."
unzip -q $ZIP_NAME -d $DATA_DIR
tar -xzf dataset.tar.gz

echo "✅ Dataset unzipped to: $DATA_DIR"
echo "🏃‍♂️ Launching training..."
python -m splitter.train

