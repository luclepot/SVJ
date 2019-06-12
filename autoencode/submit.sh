#!/bin/bash
echo 'RUNNING NOW'

env -i HOME=$HOME bash -i -c "source /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/autoencode/setup.sh
python /afs/cern.ch/work/l/llepotti/private/CMS/CMSSW_8_0_20/src/autoencodeSVJ/autoencode/autoencode.py"
