# autoencodeSVJ

autoencoders + semivisible jet analysis implementation

# usage

### selection/conversion

this analysis is implemented in sections, the first two of which are controlled by the `driver.py` file in the main repository. <br>
to see general driver behavior, run `python driver.py -h` using any old python installation.

your options are 
 
 - `select`: select events from a delphes root file, use -h for more info
 - `convert`: convert selections into h5 files (trainable data)

### training

training is implemented entirely in the `autoencode` directory. Here there is a helper module, `autoencodeSVJ`, whose components can be used in python code analyses. A few current examples of such analyses (jupyter notebooks!) are given in the `autoencode/notebooks directory`. 

The environment used for training is python `2.7.*`, with keras/tensorflow. On lxplus you might use the provided `setup.sh`. 
