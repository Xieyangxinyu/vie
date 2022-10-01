# Variable Importance Estimation Package
This is a simple and unified package for nonlinear variable importance estimation that incorporates uncertainty in the prediction function and is compatible with a wide range of machine learning models (e.g., tree ensembles, kernel methods, neural networks, etc). Below please find the detailed explanation of the package resources.

* Github repo: https://github.com/wdeng5120/vie
* PyPi: https://pypi.org/project/vie/0.0.2/

## Prerequisites
For Windows, it is recommended to run this app on a Linux emulation layer such as the Git Bash terminal. See the "Instructions for Git Bash" section for details. In addition to Git Bash, make sure you also have Python3 and Pip3 as described below.

For Mac and Linux, this app should work out of the box on the Linux or Mac terminal, but make sure you also have Python3 and Pip3 as described below.

Requirements:

* Python3 (version 3.7 or greater) - Install Python3 here: [https://www.python.org/downloads/]. Check version with: ```python3 --version```.
* Pip3 (version 20.2.1 or greater) - Make sure to install python3-pip in order to use pip install. Check version with: ```pip3 --version```.

## Installation
There are a couple of options to install this app:

* Pip Install - This app is hosted on PyPi and can be installed with the following command:
```
pip3 install vie
```
* Local Install - Alternatively, you can download or git clone the Github repo and install it locally with the following:
```
git clone https://github.com/wdeng5120/vie.git
cd vie
pip3 install -e .
```
To uninstall this app:
```
pip3 uninstall vie
```
* If you used the local install option, you will also want to delete the ```.egg-info``` file located in the ```vie/``` directory of the package. This gets created automatically with ```pip3 install -e .```.

## Usage
We provide a colab tutorial (tutorial.ipynb) on how to use the package.


