'''
Created on Aug 13, 2014

@author: GamulinN
'''
from datahandler import DataHandler

import pylab as pl
import numpy as np
import datetime
from matplotlib import pyplot
from dateutil.parser import parse

class DataVisualization(object):
    
    @staticmethod
    def barPlotTemporalData(x, y, startDate='2011-01-01', endDate='2012-01-01'):
        itemStart = datetime.datetime.strptime(startDate, "%Y-%m-%d %H:%M:%S")
        itemEnd = datetime.datetime.strptime(endDate, "%Y-%m-%d %H:%M:%S")
        if itemStart < x[0]:
            indexStart = 0
        else:
            indexStart = x.index(itemStart)
        if itemEnd > x[-1]:
            indexEnd = len(x) -1
        else:
            indexEnd = x.index(itemEnd) 
            
        #test
        #indexStart = 10
        #indexEnd = 60
           
        x = x[indexStart:indexEnd + 1]
        y[0] = y[0][indexStart:indexEnd + 1]
        y[1] = y[1][indexStart:indexEnd + 1]
        fig = pl.figure()
        #http://stackoverflow.com/questions/11617719/how-to-plot-a-very-simple-bar-chart-python-matplotlib-using-input-txt-file
        width = .5
        ind = np.arange(len(x))
        pl.bar(ind, y[0], width=width, color = 'green', alpha = 0.5)
        pl.bar(ind, y[1], width=width, color = 'yellow', alpha = 0.5)
        #pl.bar([0, 20, 50], [40, 60, 120], width=width, color = 'red')
        pl.xticks(ind + width / 2, x)

        fig.autofmt_xdate()
        pyplot.show()

if __name__ == '__main__':
    [X, y] = DataHandler.getTrainingData()
    x = X[0]
    dates = [datetime.datetime.strptime(item, "%Y-%m-%d %H:%M:%S") for item in X[0]]
    DataVisualization.barPlotTemporalData(dates, y, '2011-01-01 16:00:00', '2011-01-03 15:00:00')