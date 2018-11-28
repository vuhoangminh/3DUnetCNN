from __future__ import print_function
import importlib
from distutils.version import LooseVersion

import os, sys, inspect

# check that all packages are installed (see requirements.txt file)
required_packages = {'jupyter', 
                     'numpy',
                     'matplotlib',
                     'ipywidgets',
                     'scipy',
                     'pandas',
                     'SimpleITK'
                    }

problem_packages = list()
# Iterate over the required packages: If the package is not installed
# ignore the exception. 
for package in required_packages:
    try:
        p = importlib.import_module(package)        
    except ImportError:
        problem_packages.append(package)
    
if len(problem_packages) is 0:
    print('All is well.')
else:
    print('The following packages are required but not installed: ' \
          + ', '.join(problem_packages))

import SimpleITK as sitk
import glob
print(sitk.Version())

print (os.getcwd())
brats_folder = os.path.join(os.getcwd(), "data\\original")
print(glob.glob(os.path.join(brats_folder, "*", "*")))
print(os.path.join(brats_folder, "*", "*"))

print(glob.glob("\\pfs\\nobackup\\home\\m\\minhvu\\3DUnetCNN\\brats\\data\\original\\*\\*"))





""" path = "C:\\Users\\minhm\\Documents\\GitHub\\3DUnetCNN\\"
os.chdir(path)
print (os.getcwd())

sys.path.insert(0, path) """

# from brats.preprocess import convert_brats_data
# print("Start preprocessing...")
# convert_brats_data("\\data\\original", "\\data\\preprocessed")