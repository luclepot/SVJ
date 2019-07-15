SOURCE_DIR="/cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh"

echo "assuming platform lxplus"

# if no source dir, use t3 assumption
if [ ! -f $SOURCE_DIR ]; then
    echo "False assumption; assuming now t3 for platform"
    SOURCE_DIR="/work/llepotti/python27_tensorflow_cpu/setup.sh"
fi 

echo "source-ing file at dir $SOURCE_DIR"
source $SOURCE_DIR

# source delphes if exists
if [ -f $DELPHES_DIR/delphes-env.sh ]; then
    source $DELPHES_DIR/delphes-env.sh
fi
echo "Building autoencodeSVJ module"
source build.sh


