# Math Foundations in Machine Learning - Final Project

This is the main piece of code used for this project.
It creates a Neural-Network model, according to a given input file,
trains it and logs its performance to external files.

## Prerequisites

This code is based on Python 3, using:
 - Keras
 - TensorFlow (as backend),
 - Numpy
 - Pandas, for saving layers outputs as csv files.
 - boto3
 
 and their respectful dependecies.
 
### Installing dependencies
 
 Each package may be installed using PIP from terminal:
```
pip3 install package_name
```
or using Anaconda (from the conda environment terminal)
```
conda install package_name
```

Some installation may require admin privilege.
See each package dependecies for further explanations.

## Running this code
Running may be performed using terminal commands:
```
python3_path main.py_relative_path --[experiment_config_file_relative_path]
```
or
```
python3_path main.py_relative_path -i [experiment_config_file_relative_path]
```

For example, if python3 is an environment variable in windows, and the terminal is in the project's main directory
 (MathFoundationsInML_FinalProject), one may run the program using:
 ```
python3 main.py --[experiment_file.json]
```
or
 ```
python3 main.py -i [experiment_file.json]
```

For more instructions, one may run the help commands:
```
python3_path main.py_relative_path --help
```
or
```
python3_path main.py_relative_path -h
```

The experiment configuration file is optional. If not given as input,
 the file [experiment_config.json](experiments_config.json) will be used instead.

## Authors

* **Elad Eatah**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **Keras Team** - for the usage of  Keras in this project.
* **TensorFlow Team** - for using TensorFlow as backend. TensorFlow, the TensorFlow logo and any related marks are trademarks of Google Inc.
* **Amazon Inc.**- for usage of boto3.
* **Yu-Yang**- whose [code](https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure/48393723#48393723)
 used for plotting Training and validation in single graph to TensorBoard.


