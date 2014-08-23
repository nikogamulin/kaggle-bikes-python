'''
Created on Aug 10, 2014

@author: GamulinN
'''

from collections import defaultdict
from dateutil.parser import parse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import numpy as np
import evaluation
from sklearn.cross_validation import KFold

from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest

from datahandler import DataHandler

def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False) #center=True (the default) would not work ("ValueError: center=True only allowed for dense data") but should presumably work in general

if __name__ == '__main__':
    [X, y] = DataHandler.getTrainingData()
    X = DataHandler.getFeatures(X)
    
    yCasual = y[0]
    yRegistered = y[1]
    
    kf = KFold(len(X), n_folds=10)
    scoresCasualExtraTreesRegression = []
    scoresRegisteredExtraTreesRegression = []
    scoresTotalExtraTreesRegression = []
    
    scoresCasualABR = []
    scoresRegisteredABR = []
    scoresTotalABR = []
    
    mdlExtraTreesRegressorCasual = None
    mdlExtraTreesRegressorRegistered = None
    
    for train, test in kf:
        XTrain = X[train]
        yTrainCasual = yCasual[train]
        yTrainRegistered = yRegistered[train]
        
        XTest = X[test]
        yTestCasual = yCasual[test]
        yTestRegistered = yRegistered[test]
    
        clfCasual = ExtraTreesRegressor()
        #selectorCasual = SelectKBest(score_func=f_regression,k=20)
        #print [1+zero_based_index for zero_based_index in list(selectorCasual.get_support(indices=True))]
        clfRegistered = ExtraTreesRegressor()
        modelCasualETR = clfCasual.fit(XTrain, yTrainCasual)
        modelRegisteredETR = clfRegistered.fit(XTrain, yTrainRegistered)
        
        predictionsCasual = clfCasual.predict(XTest)
        actualCasual = yCasual[test]
        scoresCasualExtraTreesRegression.append(evaluation.rmsle(actualCasual, predictionsCasual))
        
        predictionsRegistered = clfRegistered.predict(XTest)
        actualRegistered = yRegistered[test]
        scoresRegisteredExtraTreesRegression.append(evaluation.rmsle(actualRegistered, predictionsRegistered))
        
        predictionsTotal = np.sum([predictionsCasual, predictionsRegistered], axis=0)
        actualTotal = np.sum([yTestCasual, yTestRegistered], axis=0)
        
        currentScore = evaluation.rmsle(actualTotal, predictionsTotal)
        scoresTotalExtraTreesRegression.append(evaluation.rmsle(actualTotal, predictionsTotal))
        if currentScore == min(scoresTotalExtraTreesRegression):
            mdlExtraTreesRegressorCasual = modelCasualETR
            mdlExtraTreesRegressorRegistered = modelRegisteredETR
            
        clfAdaBoostCasual = RandomForestRegressor()
        clfAdaBoostRegistered = RandomForestRegressor()
        
        modelCasualABR = clfAdaBoostCasual.fit(XTrain, yTrainCasual)
        modelRegisteredABR = clfAdaBoostRegistered.fit(XTrain, yTrainRegistered)
        
        predictionsCasualABR = clfAdaBoostCasual.predict(XTest)
        predictionsRegisteredABR = clfAdaBoostRegistered.predict(XTest)
        scoresCasualABR.append(evaluation.rmsle(actualRegistered, predictionsCasualABR))
        scoresRegisteredABR.append(evaluation.rmsle(actualRegistered, predictionsRegisteredABR))
        
        predictionsTotalABR = np.sum([predictionsCasualABR, predictionsRegisteredABR], axis=0)
        actualTotal = np.sum([yTestCasual, yTestRegistered], axis=0)
        
        currentScoreABR = evaluation.rmsle(actualTotal, predictionsTotalABR)
        scoresTotalABR.append(evaluation.rmsle(actualTotal, predictionsTotalABR))
        if currentScoreABR == min(scoresTotalABR):
            mdlABRCasual = modelCasualABR
            mdlABRRegistered = modelRegisteredABR
        
    print "ExtraTreesRegression Average Scores for total, casual, registered: %f, %f, %f" %(np.mean(scoresTotalExtraTreesRegression), np.mean(scoresCasualExtraTreesRegression), np.mean(scoresRegisteredExtraTreesRegression))
    print "Best models score: %f" % min(scoresTotalExtraTreesRegression)
    
    print "Random forest Average Scores for total, casual, registered: %f, %f, %f" %(np.mean(scoresTotalABR), np.mean(scoresCasualABR), np.mean(scoresRegisteredABR))
    print "Best models score: %f" % min(scoresTotalABR)
    
    #DataHandler.storeResults(mdlExtraTreesRegressorCasual, mdlExtraTreesRegressorRegistered, 'predictionsExtraTreesRegressor.csv')
    