# General `autoencodeSVJ` module instructions

Operating instructions for BDT / data loader helper functions

</br>

## Setting up the python environment

Navigate to the repository head directory, and then do

```Bash
source default.sh
```

OR

```Bash
source autoencode/setup.sh
```

</br>

## Getting on the correct branch

I usually do work on the `develop` branch of this repository, and then merge it into the `master` branch when I have working changes. For this reason, it is best to use the `develop` branch, and frequently reload the module (explained below). 

To get onto the `develop` branch, simply run 

```Bash
git checkout develop
```

To check which branch you're on, run

```Bash
git branch
```

</br>

## Reloading the module

Setup your environment if you have not already in that session. Then navigate to the repository head directory and do 

```Bash
git pull
source autoencode/build.sh
```

</br>

## Starting jupyter notebook

On your **local** computer (i.e. the one you didn't ssh to), run the command

```Bash
ssh -N -f -L localhost:8888:localhost:8889 <username>@<hostname>
```

For example, if I wanted to do this on lxplus, I would run

```Bash
ssh -N -f -L localhost:8888:localhost:8889 llepotti@lxplus7.cern.ch
```

This sets up a tunnel between your local computer and the server you are on, over which the server can forward jupyter notebook information. 

Next, run the following on your **remote** server:

```Bash
jupyter notebook --no-browser --port=8889
```

This will start a jupyter server, which will take over the terminal. To stop this server, use the `Control + C` stopping command.

Lastly, to access the server from your host computer, go into a web browser and navigate to `localhost:8888/tree`. This will take you to an interactive screen showing the directory you were in when you ran the 'jupyer notebook' command above. You can navigate around from there, and create a new notebnook using `new -> Python2`. You can also experiment with other types of notebooks, such as the `ROOTC++` notebook. 

</br>

## Using the Python module

Now we're at the last step (*finally*). In your python environment (i.e. within a running jupyter notebook, or a python shell), run the command

```Python
import autoencodeSVJ.utils as utils
```

This will import all of the utilities from the autoencodeSVJ module under the name `utils`. From there, you can mostly just use the `utils.BDT_load_all_data` function, which will load the data and return it in the format `(X,Y), (X_test,Y_test)`.

This function is well documented, so to see more information simply run `help(utils.BDT_load_all_data)`. This will print a helpful message explaining the function, all of its arguments, and its return values.
