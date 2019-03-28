import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class LogReg:

    def __init__(self):
        pass
    
    def score(self,Y1,Y2=0):
        if str(type(Y2)) != "<class 'numpy.ndarray'>":
            Y2 = self.last_predicted
        Y1 = Y1.reshape((Y1.shape[0],1))
        Y2 = Y2.reshape((Y2.shape[0],1))
        result = np.mean(Y1==Y2)
        return result
    
    def cross_entropy_error(self,T,Y):
        err = 0
        for i in range(len(T)):
            if T[i] == 0:
                err -= np.log(1-Y[i])
            elif T[i] == 1:
                err -= np.log(Y[i])         
        return err
    
    def sigmoid(self,w,x):
        val = 1 / (1 + np.exp( -( np.dot(x,w) ) ) )
        return val
    
    def fit(self,X,T,learning_rate=0.01,h=0.001,regularization='l2',penalty=0.1):
        X = np.append(X,np.ones(X.shape[0]).reshape((X.shape[0],1)),axis=1)
        N,D = X.shape
        w = np.random.randn(D).reshape((D,1))
        
        self.cer = []
        
        while True:
            Y = self.sigmoid(w,X)
            self.cer.append(self.cross_entropy_error(T,Y))
            f = w if regularization == 'l2' else np.sign(w) if regularization == 'l1' else 0
            w -= learning_rate*(np.dot(X.T,(Y-T)) + penalty*f)
            if len(self.cer) > 2:
                if abs(self.cer[-2] - self.cer[-1]) < h:
                    break
            
        self.coef_ = w[:-1,0]
        self.intercept_ = w[-1,0]
        self.weights = w
        
    def predict(self,X):
        X = np.append(X,np.ones(X.shape[0]).reshape((X.shape[0],1)),axis=1)
        self.last_predicted = np.round(self.sigmoid(self.weights,X))
        return self.last_predicted
            
    
class ANNClassifier:
    
    def __init__(self):
        pass
    
    def score(self,Y1,Y2=0):
        if str(type(Y2)) != "<class 'numpy.ndarray'>":
            Y2 = self.last_predicted
            
        assert(Y1.shape[0]==Y2.shape[0])
        
        if Y1.shape[1] == 1:
            enc = OneHotEncoder(categories='auto')
            Y1 = enc.fit_transform(Y1).toarray()
        if Y2.shape[1] == 1:
            enc = OneHotEncoder(categories='auto')
            Y2 = enc.fit_transform(Y2).toarray()
            
        result = np.mean(np.argmax(Y1,axis=1) == np.argmax(Y2,axis=1))
        return result
    
    def sigmoid(self,X):
        val = 1 / (1 + np.exp( -(X) ) )
        return val

    def softmax(self,X):
        X = np.exp(X)
        dsum = np.sum(X, axis=1).reshape((X.shape[0],1))
        X = X/dsum
        return X
    
    def forward(self,X,w1,b1,w2,b2,activation='sigmoid'):
        Z = np.dot(X,w1) + b1
        Z = np.tanh(Z) if activation == 'tanh' else self.sigmoid(Z)
        A = np.dot(Z,w2) + b2
        A = self.softmax(A)
        return Z, A
    
    def dJ_w2(self,T,Y,Z):
        return np.dot(Z.T,(T-Y))
    
    def dJ_b2(self,T,Y):
        return np.sum((T-Y),axis=0)
    
    def dJ_w1(self,T,Y,Z,X,w2,activation='sigmoid'):
        if activation == 'sigmoid':
            ret = (X.T).dot(np.dot((T-Y),w2.T) * Z * (1-Z))
        else:
            ret = (X.T).dot(np.dot((T-Y),w2.T) * (1-Z**2))
            
        return ret
    
    def dJ_b1(self,T,Y,Z,w2):
        return np.sum((np.dot((T-Y),w2.T) * Z * (1-Z)),axis=0)
    
    def error(self,T,Y):
        if T.shape[1] == 1:
            enc = OneHotEncoder(categories='auto')
            T = enc.fit_transform(T).toarray()
            
        return np.sum(np.log(Y)*T*-1)
    
    def fit(self,X,T,M=3,activation='sigmoid',learning_rate=0.001,epochs=1000,show=False,every=100):
        N,D = X.shape
        
        if T.shape[1] == 1:
            enc = OneHotEncoder(categories='auto')
            T = enc.fit_transform(T).toarray()
            
        K = T.shape[1]
        
        self.cer = []
        
        w1 = np.random.randn(D,M)
        b1 = np.random.randn(M)
        w2 = np.random.randn(M,K)
        b2 = np.random.randn(K)

        for i in range(epochs):
            
            Z, Y = self.forward(X,w1,b1,w2,b2,activation)
            
                
            error = self.error(T,Y)
            
            if show==True and i%100==0:
                print("Error: "+str(error))
            
            self.cer.append(error)
            
            w2 += learning_rate*self.dJ_w2(T,Y,Z)
            b2 += learning_rate*self.dJ_b2(T,Y)
            w1 += learning_rate*self.dJ_w1(T,Y,Z,X,w2,activation)
            b1 += learning_rate*self.dJ_b1(T,Y,Z,w2)
            
            
        self.W2 = w2
        self.W1 = w1
        self.B2 = b2
        self.B1 = b1
        
        
        
    def predict(self,X,activation='sigmoid'):
        temp, self.last_predicted = self.forward(X,self.W1,self.B1,self.W2,self.B2,activation='sigmoid')
        return np.argmax(self.last_predicted,axis=1).reshape((self.last_predicted.shape[0],1))