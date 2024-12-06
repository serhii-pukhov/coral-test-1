#!/bin/bash

# Step 0: Downgrade Python to version 3.9
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
# these 3 lines should be added to .bashrc !!!
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

pyenv install 3.9.10
pyenv global 3.9.10

python -m venv coral_env
source "coral_env/bin/activate"

# Step 1: Install PyCoral and TFLite libraries for Python 3
# this step will fail if python version is not 3.9
pip install wheels/*

# Step 2: Add the Debian package repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add Google's apt key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list
sudo apt-get update

# Step 3: Install the Edge TPU runtime (Standard version)
sudo apt-get install libedgetpu1-std

# Optional: Install the runtime with maximum operating frequency (if required)
# sudo apt-get install libedgetpu1-max

# Step 4: Install pip dependencies for the script
pip install requests Pillow==9.5.0 numpy==1.26.4
