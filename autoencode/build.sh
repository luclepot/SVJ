#!/bin/bash
HEAD_DIR=$(git rev-parse --show-toplevel)
CUR_DIR=$(pwd)

cd "$HEAD_DIR/autoencode/module";
python setup.py install --user;
cd $CUR_DIR;
