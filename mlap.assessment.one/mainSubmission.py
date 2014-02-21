'''
Created on 15 Feb 2014

@author: Y6187553
'''
import sys
from itertools import starmap,izip
from operator import mul
from collections import namedtuple
from scipy.optimize import fmin_bfgs

#===============================================================================
# Section with common structures used throughout the code.
#===============================================================================

"""A named tuple to hold data information containing biased volume and price.
"""
Data = namedtuple('Data', ['volume','price'])

"""A named tuple to hold actual value and feature list. """
ValueFeaturesTuple = namedtuple('ValueFeaturesTuple', ['value', 'featureList'])

#===============================================================================
# Section with utility functions.
#===============================================================================

def read_csv_file(input_filename): #TODO: Change this to be more efficient
    """Read data from a csv file.
    Return the data in a structure
    containing volume and price.
    Access them as follows

    Data.volume

    Data.price
    """
    
    data = Data([], [])

    IFile = open(input_filename, 'r')

    for line in IFile:
        line = line.strip()
        row = line.split(',')

        data.volume.append(float(row[0]))
        data.price.append(float(row[1]))

    IFile.close()
    return data

def expand_data_features(data, dataStart=12, dataEnd=20):
    """
    Returns a ValueFeaturesTuple containing actual value
    and features associated with it.
    """
    if(dataStart - 10 < 0):
        raise ValueError("No Data available for the previous 10 entries.")

    values = []
    features = []
    entriesAdded = 0
    for i in range(dataStart, dataEnd):
        values.append(data.price[i])
        features.append([])
        
        featuresAdded = 0
        for j in range(i - 11, i -1):
            features[entriesAdded].append(data.price[j])
            featuresAdded += 1
        entriesAdded += 1
        
    return ValueFeaturesTuple(values, features)

def initialise_theta(size):
    """Initialises a Theta vector of a given size"""
    return [0.0]*size

def dot_product(vectorA, vectorB):
    """Calculate the dot product of two vectors by using
    efficient iteration tools library built in python.
    Pairs the vector values,
    multiplies them,
    returns their sum.
    """
    
    if len(vectorA)!=len(vectorB):
        raise ValueError("The length of both arrays must be the same.")

    return sum(starmap(mul, izip(vectorA, vectorB)))
 
def squared_loss(theta, actual, features):
    """Loss function to be used.
    Computes square loss.
    """
    hypothesis = dot_product(features, theta)
    return (actual - hypothesis)**2
#===============================================================================
# Main code section.
#===============================================================================

def linear(inputFileName):
    """Computes the linear regression
    given a the set of data provided"""
    
    rawData = read_csv_file(inputFileName) # read the raw data
    valueFeatures = expand_data_features(rawData) # get actual result-features pair
    theta = initialise_theta(len(valueFeatures.featureList[0])) # initial values for theta based
    
    xopt, fopt, gradient, bopt, funcCalls, gradCalls  = fmin_bfgs(squared_loss, theta, args=(rawData.price, valueFeatures))
    
    
    print "Done"
    
if __name__ == "__main__":
    if (len(sys.argv) > 1):
        print('Processing ...  '  + str(sys.argv) )
        linear(sys.argv[1])
    else:
        print('Input data file not given, using stock_price.csv')
        linear(".\stock_price.csv")
    
    
    
    
    
    
    
    
    