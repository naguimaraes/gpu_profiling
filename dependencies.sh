#!/bin/bash

# This script installs the dependencies for the power_profile script. 

# Install the required packages
sudo apt-get install python3
sudo apt-get install -y python3 python3-pip
sudo apt-get install -y bc
pip3 install pandas
pip3 install matplotlib
pip3 install numpy
sudo apt-get update