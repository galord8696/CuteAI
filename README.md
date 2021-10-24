# CuteAI

CuteAI is a visual Python environment for exploring machine learning.

*Currently the only integrated models are the sinusoid GAN and MNIST GAN*

## Installation
Please install the latest version of python.

Use the package manager [Anaconda](https://pip.pypa.io/en/stable/) or the lite version [Miniconda](https://docs.conda.io/en/latest/miniconda.html/) for the following:

Create a conda environment
```bash
$ conda create --name CuteAI
$ conda activate CuteAI
```

Install dependencies
```bash
$ conda install -c pytorch pytorch=1.4.0
$ python -m pip install ttkbootstrap
```

**Make sure the environment is active before you run it!**

```
$ conda activate CuteAI
$ python CuteAI.py
```
## Usage
This is a prototype. Please reload every run.
```
$ python CuteAI
```

## Resources
built in tkinter and [ttkbootstrap](https://ttkbootstrap.readthedocs.io/en/latest/index.html#) for the GUI

base for drag and drop sourced from [python 3.11.0 alpha](https://github.com/python/cpython/blob/main/Lib/tkinter/dnd.py)

ML models sourced from [RealPython](https://realpython.com/generative-adversarial-networks/#author)

Cute images sourced from [Berkeley AI](http://ai.berkeley.edu/)