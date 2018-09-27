#!/bin/bash

for folder in $(ls -d */); do
   echo "Testing ${folder%%/} to see if it works correctly."; echo ""
   cd $folder
   ipython --matplotlib=osx ${folder%%/}.py
   echo "-------------------"
   echo ""
   cd ..
done
