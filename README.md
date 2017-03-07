# DeepForest

A simple implementation in python of the method proposed by Zhi-Hua Zhou and Ji
 Feng in their paper "Deep Forest: Towards An Alternative to Deep Neural 
 Networks". You can check their article [here](https://arxiv.org/pdf/1702.08835.pdf)
 <br><br>
 
 The implementation is based on the great work of the Scikit-learn library. The 
 API is built upon the scikit way of doing data science, by defining object with 
 fit and predict methods. Some example will be provided soon...
 
 
 ## Installation
 
 The recommended way to proceed for the installation is through conda environments.
 Using conda you need to type :
 ```bash
conda create -n deepforest python=3
source activate deepforest
conda install --file requirements.txt
python setup.py install
```