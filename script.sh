#!/bin/bash
module load libraries/cuda-7.5
make
python bench.py
make clean
