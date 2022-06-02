#!/bin/bash
# Run python script in specified folder, passing all cli arguments
# Usage: exec.sh <FOLDER> <PYTHON_FILE> <PARAMETERS>

cd $1
shift 1 # remove first cli parameter
python $@