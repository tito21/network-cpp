#!/bin/sh


g++ network.cpp -o network -std=c++11 -O2 -larmadillo

echo "Running ./network"

if [ $? -eq 0 ]
then
    ./network
fi