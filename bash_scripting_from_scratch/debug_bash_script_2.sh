#!/bin/bash

# Your script goes here
echo "HEllo"
x=12
echo "Value of variable x is: $x"

if [ $? -ne 0 ]; then
    echo "Error ocurred"
fi
