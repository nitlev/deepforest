# DeepForest
[![Build Status](https://travis-ci.org/nitlev/deepforest.svg?branch=master)](https://travis-ci.org/nitlev/deepforest)
[![Coverage Status](https://coveralls.io/repos/github/nitlev/deepforest/badge.svg?branch=dev)](https://coveralls.io/github/nitlev/deepforest?branch=dev)


## You'll find here
a simple implementation in python of the method proposed by Zhi-Hua Zhou and Ji
Feng in their paper "Deep Forest: Towards An Alternative to Deep Neural Networks".
You can check their article [here](https://arxiv.org/pdf/1702.08835.pdf)
<br><br>
 
The implementation is based on the great work of the Scikit-learn library. The
API is built upon the scikit way of doing data science, by defining object with
fit and predict methods. Some example will be provided soon! In the meantime, 
you may start by checking up the notebooks.
 
 
## Installation
 
The recommended way to proceed for the installation is through conda environments.
Using conda you need to type :
```bash
conda create -n deepforest python=3
source activate deepforest
conda install --file requirements.txt
python setup.py install
```