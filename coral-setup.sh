#!/bin/bash

# Step 1: Add the Debian package repository
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

# Add Google's apt key
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update the package list
sudo apt-get update

# Step 2: Install the Edge TPU runtime (Standard version)
sudo apt-get install libedgetpu1-std

# Optional: Install the runtime with maximum operating frequency (if required)
# sudo apt-get install libedgetpu1-max

# Step 3: Install PyCoral library for Python 3
sudo apt-get install python3-pycoral

# Step 4: Install dependencies to run the ML loading script
# TODO pip3 install requests Pillow

#mkdir coral && cd coral
#git clone https://github.com/serhii-pukhov/coral-test-1.git
#cd coral-test-1
#python3 run.py