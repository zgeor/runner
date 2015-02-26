#!/usr/bin/python
'''
Created on 26 Feb 2015

@author: Zhivko Georgiev
'''

#####################################
# User imports
#####################################
import sys
import argparse
import re
import csv
import string
import numpy as numpy
from scipy import integrate, interpolate
from collections import namedtuple
from math import sin 
from operator import itemgetter
from matplotlib.pylab import *
#####################################
# Setup variables
#####################################
ModelParsingRegExpGroup = r'(.*)\' = (.*){MSE = ((\d*\.\d*)|inf|-inf), MDL = ((\d*\.\d*)|inf|-inf)}$'
ModelMeasurmentMatcher = re.compile(ModelParsingRegExpGroup)
#####################################
# Global variables
#####################################

#####################################
# Custom structures
#####################################

#####################################
# Functional code
#####################################
def prepareData(fileName):
    """
    Remark          Very bad implementation. 
                    Could be improved by using iter() and reading file
                    in chunks.
                    
    This method reads the data file into list of lists
    
    returns List<List<str>>, int
    """
    allRows = []
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            rowEnd = row.pop().rstrip(';')
            row.append(rowEnd)
            allRows.append(row)
    
    for x in range(1, len(allRows)):
        allRows[x] = [float(i) for i in allRows[x]]       
    return allRows, len(allRows)

def preparedDataRow(dataLists):
    """
    Gets dictionary of column names and row values.
    
    returns dict<str, float>, int
    """
    for i in range(1, len(dataLists)):
        yield dict(zip(dataLists[0], dataLists[i]))

def evaluateModel(model, variables):
    """
    Evaluates a model given all datapoints required.
    
    LModel     model         the model to be evaluated
    dict       variables     the variables of the model
    
    Remark    Need to restrict the global values used for evaluation.
    
    return float
    """
    return eval(model.replace('variable_', ''), globals(), variables)

def trapezoidalIntegration(calculated, actual, timeStep):
    """
    Reverse engineered from lagramge 1.2r simulate.c.
    This is in fact trapezoidal integration with addition of initial value.
    
    """
    i = 0
    output = numpy.zeros((actual.size, ))
    
    summation = output[0] = actual[0]
    
    for i in range(1, actual.size):
        summation += (calculated[i -1] + calculated[i])* timeStep / 2
        output[i] = summation 
    return output

def adamBashforth2Integration(calculated, actual, timeStep):
    """
    Implementation of Adam-Bashforth 2 integration.
    """
    output = numpy.zeros((actual.size, ))
    summation = 0 
    output[0] = actual[0]
    output[1] = actual[1]
    
    summation += actual[1]
    for i in range(2, actual.size):
        summation += ((3/2)*calculated[i-1] - (1/2)*calculated[i-2])* timeStep
        output[i] = summation 
    return output

def rateModel(eq, pVarName, dataFileName, timeStep):
    """
    Rates the Lagramge model, according to some Accuracy results.
    
    returns float, float
    """
    preppedData, dataLength = prepareData(dataFileName)
    
    mse = 0.0
    mpe = 0.0
    
    evaluationDataPoints = 0.0
    if timeStep:
        calculated = numpy.zeros((dataLength - 1, ))
        for data in preparedDataRow(preppedData):
            calculated[evaluationDataPoints] = evaluateModel(eq, data)
            evaluationDataPoints += 1
                
        actual = numpy.array(map(itemgetter(preppedData[0].index(pVarName)), preppedData[1:dataLength]))
        
        predicted = adamBashforth2Integration(calculated, actual, timeStep)
        #predicted = trapezoidalIntegration(calculated, actual, timeStep)
        
        error = numpy.subtract(actual, predicted)
        squaredError = numpy.multiply(error, error)
        mpe = numpy.average(numpy.divide(error, actual)) * 100.0
        mse = numpy.average(squaredError)
    else:
        for data in preparedDataRow(preppedData):
            evaluationDataPoints += 1
            res = evaluateModel(eq, data)
            se += calcSquaredError(data[pVarName], res)
            mpe += calcPercentageError(data[pVarName], res)

    return mse, mpe

def calcSquaredError(actualResult, forecastResult):
    """
    Calculate squared error.
    returns float
    """
    
    return  (actualResult - forecastResult)**2

def calcPercentageError(actualResult, forecastResult):
    """
    Calculates Percentage Error. 
    
    returns float
    """
    return ((actualResult - forecastResult)/actualResult) * 100

def calcAbsolutePercentageError(actualResult, forecastResult):
    """
    Calculates Absolute Percentage Error.
    """
    return (abs((actualResult - forecastResult)/actualResult)) * 100

#####################################
# Grammar functions
#####################################
def monod(c, v):
    return v / (v + c)

def step(c, v):
    if c < v:
        return v
    return 0

def ln(c):
    return log(c)

def step(c, v):
    if c < v:
        return v
    return 0

def parseProgramArgs(args):
    """
    Parses the input arguments of the program.
    Overwrites the local configuration file.
    
    returns dict<dict>
    """

    parser = argparse.ArgumentParser(description='Run LAGRAMGE equation validation process.')
    
    parser.add_argument('dataFile', metavar='lagramge data file', type=str, nargs='?',
                       help='The data file to be used for validation process.')
    
    parser.add_argument('-e', dest='eq', default=False, type=str, help='Lagramge equation')
    parser.add_argument('-t', dest='timeStep', default=False, type=float32, help='Time step for differential equations. By default ')
    
    parsed = parser.parse_known_args()

    return parsed[0].dataFile, parsed[0].eq, parsed[0].timeStep

def main(argv):
    """
    Main method that unfortunately is way too complicated.
    """    

    dataFileName, eq, diffsOnly = parseProgramArgs(argv)
    search = ModelMeasurmentMatcher.search(eq)
    group = search.group(1, 2, 3)
    dependentVar = group[0]
    eq = group[1]
    lMse = group[2]
    
    mse, mpe = rateModel(eq, dependentVar, dataFileName, diffsOnly)
    
    print "Lagramge MSE: %s" % lMse
    print "MSE: %.6f" % mse
    print "MPE: %.6f" % mpe

if __name__ == '__main__':
    main(sys.argv[1:])