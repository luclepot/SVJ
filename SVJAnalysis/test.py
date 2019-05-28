import os 

path = os.path.abspath(os.path.dirname(__file__))

os.system("cd {}; cd ../../..; ls".format(path))
