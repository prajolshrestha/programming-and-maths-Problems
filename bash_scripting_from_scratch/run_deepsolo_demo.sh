#!/bin/bash

cd ${HOME}/DeepSolo
ls
python --version #conda activate deepsolo
echo "We are inside code base and all set to visualize a map data."

cd adet/data/datasets
python ./testmap3.py