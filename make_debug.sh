#!/bin/sh


g++ network.cpp -o network -g -std=c++11 -O2 -larmadillo

if [ $? -eq 0 ]
then
    ./network
fi
