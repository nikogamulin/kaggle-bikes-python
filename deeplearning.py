'''
Created on 3. sep. 2014

@author: niko
'''

from datahandler import DataHandler
from sklearn.cross_validation import KFold
from sklearn.neural_network import BernoulliRBM
from sklearn import datasets, linear_model
import evaluation

import numpy as np

if __name__ == '__main__':
    [X, y] = DataHandler.getTrainingData()
    X = DataHandler.getFeatures(X)
    X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
    
    yCasual = y[0]
    yRegistered = y[1]
    
    kf = KFold(len(X), n_folds=10)
    for train, test in kf:
        XTrain = X[train]
        yTrainCasual = yCasual[train]
        yTrainRegistered = yRegistered[train]
        
        XTest = X[test]
        yTestCasual = yCasual[test]
        yTestRegistered = yRegistered[test]
        
        model1 = BernoulliRBM(n_components=80)
        model1.fit(XTrain)
        weights_1 = model1.components_
        XTrain_stage1 = XTrain.dot(weights_1.T)
        
        model2 = BernoulliRBM(n_components=40)
        model2.fit(XTrain_stage1)
        weights_2 = model2.components_
        XTrain_stage2 = XTrain_stage1.dot(weights_2.T)
        
        #perform linear regression
        regrCasual = linear_model.LinearRegression()
        regrCasual.fit(XTrain_stage2, yTrainCasual)
        
        regrRegistered = linear_model.LinearRegression()
        regrRegistered.fit(XTrain_stage2, yTrainRegistered)
        
        predictionsCasual = regrCasual.predict(XTrain_stage2)
        score = evaluation.rmsle(yTrainCasual, predictionsCasual)
        print score
        