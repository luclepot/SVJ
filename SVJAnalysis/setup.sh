#!/bin/bash
alias svj="python $(pwd)/SVJAnalysis.py $@"
alias svjtest="python $(pwd)/SVJAnalysis.py $@ -i $(pwd)/intest -o $(pwd)/outtest -d -t -c -f qcd*"
