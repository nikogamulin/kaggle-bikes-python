'''
Created on Aug 9, 2014

@author: GamulinN
'''
import csv
#import pandas as pd
from collections import defaultdict
from dateutil.parser import parse
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

trainingDataFile = './data/train.csv'
testDataFile = './data/test.csv'

class DataHandler(object):
    '''
    classdocs
    '''
    
    @staticmethod
    def getTrainingData():
        columns = defaultdict(list) # each value in each column is appended to a list

        with open(trainingDataFile) as f:
            reader = csv.DictReader(f, delimiter = ",") # read rows into a dictionary format
            for row in reader: # read a row as {column1: value1, column2: value2,...}
                for (k,v) in row.items(): # go over each column name and value 
                    columns[k].append(v) # append the value into the appropriate list
                                         # based on column name 
        X = [columns['datetime'], columns['season'], columns['holiday'], columns['workingday'], columns['weather'], columns['temp'], columns['atemp'], columns['humidity'], columns['windspeed']]
        #y = [[np.float(i) for i in columns['casual']], [np.float(i) for i in columns['registered']] ]
        y = [np.asarray([np.float(i) for i in columns['casual']]), np.asarray([np.float(i) for i in columns['registered']])]
        return [X, y]
    
    @staticmethod
    def parseResultsFile():
        columns = defaultdict(list) # each value in each column is appended to a list

        with open(testDataFile) as f:
            reader = csv.DictReader(f, delimiter = ",") # read rows into a dictionary format
            for row in reader: # read a row as {column1: value1, column2: value2,...}
                for (k,v) in row.items(): # go over each column name and value 
                    columns[k].append(v) # append the value into the appropriate list
                                         # based on column name 
        X = [columns['datetime'], columns['season'], columns['holiday'], columns['workingday'], columns['weather'], columns['temp'], columns['atemp'], columns['humidity'], columns['windspeed']]
        return X
    
    @staticmethod
    def getFeatures(X):
        rows = []
        for i in range(0, len(X[0])):
            currentRow = []
            currentDateTime = parse(X[0][i])

            if currentDateTime.year == 2011:
                currentRow.extend([1,0])
            else:
                currentRow.extend([0,1])
                
            currentMonth = [0] * 12
            currentMonth[currentDateTime.month - 1] = 1
            currentRow.extend(currentMonth)
            
            currentDayOfWeek = [0] * 7
            currentDayOfWeek[currentDateTime.weekday()] = 1
            currentRow.extend(currentDayOfWeek)
            
            currentHour = [0] * 24
            currentHour[currentDateTime.hour] = 1
            currentRow.extend(currentHour)         
            
            s = int(X[1][i])
            currentSeason = [0] * 4
            currentSeason[s - 1] = 1
            #seasons.append(currentSeason)
            currentRow.extend(currentSeason)
            
            holiday = int(X[2][i])
            currentRow.append(holiday)
            
            workingDay = int(X[3][i])
            currentRow.append(workingDay)
            
            weather = [0] * 4
            weather[int(X[4][i]) - 1] = 1
            currentRow.extend(weather)
            
            currentRow.append(np.float(X[5][i]))
            currentRow.append(np.float(X[6][i]))
            currentRow.append(np.float(X[7][i]))
            currentRow.append(np.float(X[8][i]))
            
            #ExtraTreesRegression: 0.334495, RandomForests = 0.391659
            #ExtraTreesRegression Average Scores for total, casual, registered: 0.494197, 0.809773, 0.442037
            #Random forest Average Scores for total, casual, registered: 0.601015, 1.713724, 0.570361
            
            #additional efatures
            #afternoon, morning, evening,night, mistEightOClock,clearSeventeenOClock, clearEighteenOClock
            '''
            timeOfDay = [0] * 4
            if currentHour in [0,1,2,3,4,5,6]:
                timeOfDay[0] = 1
            elif currentHour in [7,8,9,10,11]:
                timeOfDay[1] = 1
            elif currentHour in [12,13,14,15,16,17,18]:
                timeOfDay[2] = 1
            else:
                timeOfDay[3] = 1
                
            currentRow.extend(timeOfDay)
            
            ExtraTreesRegression Average Scores for total, casual, registered: 0.502293, 0.825507, 0.446744
            Best models score: 0.343991
            Random forest Average Scores for total, casual, registered: 0.588075, 1.726128, 0.548196
            Best models score: 0.401697
            '''
            
            if X[4][i] == 1 and currentHour == 8:
                clearEightOClock = 1
            else:
                clearEightOClock = 0
                
            currentRow.append(clearEightOClock)
                
            if X[4][i] == 1 and currentHour == 18:
                clearEighteenOClock = 1
            else:
                clearEighteenOClock = 0
            
            currentRow.append(clearEighteenOClock)
            #ExtraTreesRegression Average Scores for total, casual, registered: 0.494197, 0.809773, 0.442037
            #Random forest Average Scores for total, casual, registered: 0.601015, 1.713724, 0.570361
            
            
            rows.append(currentRow)
            
        return np.array(rows)
    
    @staticmethod
    def storeResults(modelCasualETR, modelRegisteredETR, destinationFilename):
        X = DataHandler.parseResultsFile()
        features = DataHandler.getFeatures(X)
        predictionsCasual = modelCasualETR.predict(features)
        predictionsRegistered = modelRegisteredETR.predict(features)
        results = []
        for i in range(0, len(X[0])):
            datetime = X[0][i]
            totalBikesPrediction = predictionsCasual[i] + predictionsRegistered[i]
            results.append([datetime, totalBikesPrediction])
            
        with open(destinationFilename, "wb") as f:
            c = csv.writer(f, delimiter=',')
            c.writerow(['datetime', 'count'])
            for item in results:
                c.writerow(item)
        
        
                
                
            
        


if __name__=="__main__":
    [X, y] = DataHandler.getTrainingData()
    features = DataHandler.getFeatures(X)
    clf = ExtraTreesRegressor()
    clf = clf.fit(features, y[0])
    predictions = clf.predict(features)
    scores = mean_squared_error(predictions, y[0])
    print scores
    test = clf.predict(X[6])
    
        