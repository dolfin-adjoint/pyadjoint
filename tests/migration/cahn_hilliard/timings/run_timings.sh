#!/bin/bash

export PYTHONPATH=../../../..

for i in $(seq 1 5)
do
  /usr/bin/time python $1 2>&1
done
