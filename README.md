# LogisticRegressionAndBackPropPYY

LogisticRegressionAndBackPropPYY is a python utility class I wrote to implement the backpropogation algorithm. It uses softmax and backpropogation to perform multiclass classification.
A logistic regression class is also included in this utility to perform binary classification.

Although you can get better optimized results by using libraries like scikit learn, LogisticRegressionAndBackPropPYY is focussed mainly for new learners.
Feel free to edit/modify/debug/add to/play with/etc. to my code...


## Usage

Just download and put the file LogisticRegressionAndBackPropPYY.py in the directory where your python file is located.
Make sure you have numpy, pandas and sklearn installed (code just uses one hot encoder from sklearn).

It is best if you are using Anaconda/Spyder as it comes with these libraries by default.

```python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from LogisticRegressionAndBackPropPYY.py import LogReg, ANNClassifier

# X = Your training test data (should be a numpy array of shape n,m)
# Y = Your training target data
# for LogReg, Y should be a numpy array containing 0s and 1s only and of the shape n,1
# for ANNClassifier, Y can be a numpy array containing numbers from 0 to k(~ number of classes) and of the shape n,1. The class does one hot encoding by itself so you do not need to one hot encode Y

# Xtest = Your test dataset
# Ytest = Your test target dataset


# LOGISTIC REGRESSION

# create an object of the class
logistic = LogReg()

# fit the data
logistic.fit(X,Y,learning_rate=0.01,h=0.001,regularization='l2',penalty=0.1)
# you can specify the learning rate, regularization ('l1','l2','NA'), regularization penalty, and error difference (h - when the absolute error is less than h, the loop breaks out and stops)

# make predictions
Yhat = logistic.predict(Xtest)

# get prediction score
score = logistic.score(Ytest,Yhat)
# the parameter Yhat is optional to pass. If you do not pass that, the program just uses the last predicted value by default

# plot and see how the error decreased
plt.plot(logistic.cer)


# ANN CLASSIFIER

# create an object of the class
classifier = ANNClassifier()

# fit the data
classifier.fit(X,Y,M=3,activation='sigmoid',learning_rate=0.001,epochs=1000,show=False,every=100)
# you can specify the hidden layer size M, learning rate, activation function for the hidden layer ('sigmoid', 'tanh'), number of epochs, show (True/False ~ if you want to print iterations and see errors converge or not), every (errors will be printed after these many iterations)

# make predictions
Yhat = classifier.predict(Xtest,activation='sigmoid')
# usually use the same activation here that you used to train/fit the data

# get prediction score
score = classifier.score(Ytest,Yhat)
# the parameter Yhat is optional to pass. If you do not pass that, the program just uses the last predicted value by default. One hot encoding is done automatically

# plot and see how the error decreased
plt.plot(classifier.cer)

```

## Contributing
Pull requests are welcome! Please let me know if this was helpful :)
Visit my [website](https://raghavkharbanda.com) if you wish to contact me!


## License
[MIT](https://choosealicense.com/licenses/mit/)