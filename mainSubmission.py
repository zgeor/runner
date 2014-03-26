'''
Created on 15 Feb 2014

@author: Y6187553
'''
import sys
import time
from collections import namedtuple, deque
from math import fabs as abs
from scipy.optimize import fmin_bfgs, minimize
from scipy.misc import logsumexp
from numpy import array as nparray, zeros as npzeros, exp as npexp, log as nplog, loadtxt, ones as npones
from numpy import argmax as npargmax, sum as npsum, multiply as npmult, absolute as npabsolute
from numpy import asarray as npasarray, rollaxis as nprollaxis, ndarray, append as npappend, empty as npempty, abs as npabs
from numpy.random import rand as nprandom, shuffle as npshuffle
from functools import reduce, partial
#===============================================================================
# Section with common structures used throughout the code.
#===============================================================================

"""A named tuple to hold data information containing biased volume and price.
"""
Data = namedtuple('Data', ['volume','price'])

"""A named tuple to hold actual value and feature list. """
ValueFeaturesTuple = namedtuple('ValueFeaturesTuple', ['value', 'featureList'])

#===============================================================================
# Section with global variables used throughout the code.
#===============================================================================

"""Line separator """
LineSeparator = "----------------------------"

"""Maximum iterations for optimisation"""
MaxIterations = 5

"""Number of logistic regression classes initialise to -1 so if forgotten to set throw an error."""
LogisticClassCount = -1
#===============================================================================
# Section with utility functions.
#===============================================================================

def read_csv_file(input_filename): #TODO: Change this to be more efficient
    """Read data from a csv file.
    Return the data in a matrix
    """
    data = loadtxt(input_filename, delimiter=',')
    return data

def calculate_class(currentDay, previousDay):
    """
    Adds a class row to the data.
    """
    global LogisticClassCount 
    LogisticClassCount = 5
    
    logisticClass = -1
    change = 100 * (currentDay - previousDay)/previousDay
    if(abs(change) <= 5):
        logisticClass = 0
    elif(change <= 10 and change > 5):
        logisticClass = 1
    elif(change >= -10 and change <-5):
        logisticClass = 2
    elif(change > 10):
        logisticClass = 3
    elif(change < -10):
        logisticClass = 4      
    else:
        raise ValueError("Could not expand price class.")
    
    return logisticClass

def split_set(data, trainingSetSize, reverseSets, randomiseData=False):
    """Splits the input data into training and validation set.
    Shuffles the data
    ------------------
    data - ValueFeatures tuple
    
    trainingSetSize a float indicating the portion of the training set.
    
    ------------------
    returns 
    """
    if(trainingSetSize > 1) or ( trainingSetSize < 0):
        raise ValueError("The value of trainingSetSize must be in between [0:1].")
    
    dataLen = len(data[0])
    trainingSetLen = int(round(dataLen * trainingSetSize))
    evaluationSetLen = dataLen - trainingSetLen
    print LineSeparator
    print "Training dataset size: {0}\nEvaluation dataset size: {1}\nTotal dataset size: {2}".format(trainingSetLen, evaluationSetLen, dataLen)
    print LineSeparator
    
    if(randomiseData):
        print "Shuffling the rows."
        shuffledData = npappend(data.featureList, data.value.reshape(data.featureList.shape[0], 1), axis=1)
        npshuffle(shuffledData)
        data = ValueFeaturesTuple(shuffledData[:, shuffledData.shape[1]-1],shuffledData[:, :shuffledData.shape[1]-1])
        
    if(reverseSets == False):
        return ValueFeaturesTuple(data.value[:trainingSetLen],data.featureList[:trainingSetLen]),ValueFeaturesTuple(data.value[trainingSetLen:],data.featureList[trainingSetLen:])
    if(reverseSets == True):
        return ValueFeaturesTuple(data.value[trainingSetLen:],data.featureList[trainingSetLen:]), ValueFeaturesTuple(data.value[:trainingSetLen],data.featureList[:trainingSetLen])
    
def expand_data_features(data, calculate_class = None):
    """
    Returns a ValueFeaturesTuple containing actual value
    and features associated with it.
    """
    if(calculate_class != False):
        print "Info: Using class as value."
    else:
        print "Info: Using price as value."
    dataStart = 10
    dataEnd = data.shape[0]
    if(dataStart >= dataEnd):
        raise ValueError("No Data available for the previous 10 entries.")

    values = npempty(dataEnd - dataStart)
    features = []
    entriesAdded = 0
    for i in range(dataStart, dataEnd):
        if(calculate_class != None):
            values[i - dataStart] = calculate_class(data[i][1], data[i-1][1])
        else:
            values[i - dataStart] = data[i][1]
        features.append([])
        
        features[entriesAdded].append(1.0)
        features[entriesAdded].append(data[i-1][1]**4)
        features[entriesAdded].append(data[i-2][1]**3)
        features[entriesAdded].append(data[i-3][1]**2)
        features[entriesAdded].append(data[i-4][1])
        #=======================================================================
        # for j in range(i - 10, i):
        #     features[entriesAdded].append(data[j][1])
        #     features[entriesAdded].append(data.volume[j])
        #     features[entriesAdded].append(data.volume[j]**2)
        #     features[entriesAdded].append(data[j][1] - data[j-1][1]) 
        #                                   
        #     if(j == i - 1):
        #         features[entriesAdded].append((data[j][1] - data[j-1][1])**2)
        #         features[entriesAdded].append((data[j][1] - data[j-1][1])**3)
        #         features[entriesAdded].append(data[j][1]**2)
        #         features[entriesAdded].append(data[j][1]**3)
        #         features[entriesAdded].append(data[j][1]**4)
        #         features[entriesAdded].append(data[j][1]**5)
        #      
        #     if(j == i-2):
        #         features[entriesAdded].append((data[j][1] - data[j-1][1])**2)
        #         features[entriesAdded].append(data[j][1]**2)
        #         features[entriesAdded].append(data[j][1]**3)
        #         features[entriesAdded].append(data[j][1]**4)
        #     if(j == i-3):
        #         features[entriesAdded].append(data[j][1]**2)
        #         features[entriesAdded].append(data[j][1]**3)
        #     
        #     total = data[j][0]
        #     if (total != 0):
        #         features[entriesAdded].append(nplog(total))
        #     else:
        #         features[entriesAdded].append(0)
        #=======================================================================
        entriesAdded += 1
    print "Info: Total feature count: {0}".format(len(features[0]))
    return ValueFeaturesTuple(values, nparray(features))

def normalise_data(valueFeatures, isLogistic = False):
    """
    Normalises the data. 
    Skipping the 1.0 column from the feature list.
    """
    features = normalise_matrix(valueFeatures.featureList[:, 1:])
    if(not isLogistic):
        values = normalise_matrix(valueFeatures.value)
    else:
        values = valueFeatures.value
    vl = ValueFeaturesTuple(values, npappend(valueFeatures.featureList[:, :1], features, axis=1))
    return vl

def normalise_matrix(data):
    """
    Normalises a given matrix.
    """
    data = npasarray(data)
    Xr = nprollaxis(data, axis=0)

    mean = Xr.mean(axis=0)
    std = Xr.std(axis=0)
    if isinstance(std, ndarray):
        std[std == 0.0] = 1.0
    elif std == 0.:
        std = 1.
    Xr -= mean
    Xr /= std
    return Xr

def initialise_theta(a, b):
    """Initialises a Theta vector of a given size"""
    #return npzeros(shape=(a,b))
    return npones(shape=(a,b))
    #return nprandom(a,b)

def func_wrapper(f, cache_size=2):
    evals = {}
    last_points = deque()

    def get(x,y,pt, which):
        s = pt.tostring()
        if s not in evals:
            evals[s] = f(x,y,pt)
            last_points.append(s)
            if len(last_points) >= cache_size:
                del evals[last_points.popleft()]
        return evals[s][which]

    return partial(get, which=0), partial(get, which=1)

def squared_loss(theta, actual, features):
    """Loss function to be used.
    Computes square loss.
    ------------
    returns a float
    """
    hypothesis = features.dot(theta)
    loss = actual - hypothesis
    sqLoss = npsum(loss**2)
    return sqLoss

def squared_grad(theta, actual, features):
    """Loss function to be used.
    Computes square loss.
    ------------
    returns a float
    """
    hypothesis = features.dot(theta)
    loss = actual - hypothesis
    grad = features.T.dot(loss)/features.shape[0]
    return grad

def lasso_loss(theta, actual, features, weight, ftype):
    """ 
    Lasso loss function
    
    Parameters:
    theta - array of dtype=long 
    actual - actual values corresponding to the featureset
    features - the features for which the loss has been calculated
    
    weight=0.5 the lambda scalar used for regularisation
    ftype=2 the type of regularisation 
        0 - Subset selection
        1 - Lasso
        2 - Ridge
        
    returns the loss function
    """
        
    loss = weight*squared_loss(theta, actual, features) + (1.0 - weight)*(sum(npabs(theta))**ftype)
    return loss

def logistic_loss(theta, actual, features):
    """
    Logistic loss function
    """
    theta = theta.reshape(LogisticClassCount, features.shape[1])
    hypothesis = npexp(theta.dot(features.T))
    probabilities = hypothesis / npsum(hypothesis, axis = 0)
         
    jazz = npsum(reduce(npmult, nplog(probabilities)))
    return jazz

def logistic_cost(theta, actual, features):
    """
    Logistic loss function
    """
    theta = theta.reshape(LogisticClassCount, features.shape[1])
    cost = 0.0
     
    for i in range(features.shape[0]):
        dotThetaFeatures = features[i].dot(theta.T)
        logSumExp = logsumexp(dotThetaFeatures)
        est = dotThetaFeatures[actual[i]]
        cost += est - logSumExp
         
    return -1.0*cost

def reg_logistic_cost(theta, actual, features, lambdaWeight):
    """
    Logistic loss function
    """
    return lambdaWeight*logistic_cost(theta, actual, features) + (1-lambdaWeight)*npabsolute(theta).sum()

iterations = 0
def monitor_optimization(xk):
    """This function gets invoked every time the loss
    function is invoked
    """
    global iterations
    print "Iterations completed {0} \r".format(iterations),
    iterations+=1

def mean_square_error(actualValues, hypothesis):
    """
    Calculate the mean square error.
    """
    sumOfSquares =  sum((actualValues - hypothesis)**2)
    sumOfSquaresAverage = sumOfSquares / actualValues.shape[0]
    return sumOfSquaresAverage

def predict_logistic(valueFeatures, thetas):
    """
    Calculate predicted class. Hacked function. 
    Exact implementation as in the lecture slides.
    """
    result = npempty((valueFeatures.featureList.shape[0], 1))
    for i in range(valueFeatures.featureList.shape[0]):
        predictions = npempty(thetas.shape[0])
        thetaX = thetas.dot(valueFeatures.featureList[i])
        sumThetaX = npexp(npsum(thetaX))
        for j in range(thetas.shape[0]):
            predictions[j] = npexp(thetaX[j])/sumThetaX
        result[i] = npargmax(predictions, axis = 0) 
    matches = 0  
    for pVal, tVal in zip(result, valueFeatures.value):
        if(pVal == tVal):
            matches+=1
    return 100.0*matches/valueFeatures.featureList.shape[0]


def logistic_accuracy(valueFeatures, thetas):
    """
    Compute the accuracy of the logistic clasifier.
    """
    matches = 0
    predictions = npzeros((valueFeatures.featureList.shape[0], 1))
    probabilities = npzeros((valueFeatures.featureList.shape[0], thetas.shape[0]))
    for i in range(valueFeatures.featureList.shape[0]):
        probabilities[i] = sigmoid(valueFeatures.featureList[i].dot(thetas.T))
    predictions[:, 0] = npargmax(probabilities, axis = 1)
    for pVal, tVal in zip(predictions, valueFeatures.value):
        if(pVal == tVal):
            matches+=1
    return 100.0*matches/valueFeatures.featureList.shape[0]

def sigmoid(x):
    return 1 / (1 + npexp(-x))

def cross_validation(regressionFunction, inputFileName):
    """
    Cross validation function
    """    
    meanSquaredErrorOne = regressionFunction(inputFileName, False)
    meanSquaredErrorTwo = regressionFunction(inputFileName, True)
    
    average = (meanSquaredErrorOne + meanSquaredErrorTwo)/2
    print LineSeparator
    print "\tFold One: Mean SquareError {0}".format(meanSquaredErrorOne)
    print "\tFold Two: Mean SquareError {0}".format(meanSquaredErrorTwo)
    print "\tAverage: {0}".format(average)
    print LineSeparator
    pass

#===============================================================================
# Main code section.
#===============================================================================
def linear(inputFileName, reverseSets=False):
    """Computes the linear regression
    given a the set of data provided"""
    
    start_time = time.time()
    
    global MaxIterations
    global iterations
    iterations = 0
    
    print "Least squares linear regression"
    print "Maximum iterations: {0}".format(MaxIterations)
    
    rawData = read_csv_file(inputFileName) # read the raw data
    expandedValueFeatures = expand_data_features(rawData)
    
    trainingSet, evaluationSet = split_set(expandedValueFeatures, 0.5, reverseSets)
    
    trainingSetValueFeatures = normalise_data(trainingSet) # get actual result-features pair
    theta = initialise_theta(1, trainingSetValueFeatures.featureList.shape[1]) # initial values for theta based
    
    theta = fmin_bfgs(squared_loss, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList),norm=float('-inf'), disp=True, maxiter=MaxIterations, callback=monitor_optimization)
    
    evaluationSetHypothesis = evaluationSet.featureList.dot(theta)
    evaluationSetMeanSquareError = mean_square_error(evaluationSet.value, evaluationSetHypothesis)
    
    m, s = divmod((time.time() - start_time), 60)
    print "Running duration: {0:d}:{1} seconds".format(int(m), s)
    print "Least squares linear regression done\n\n"
    return evaluationSetMeanSquareError

def reglinear(inputFileName, reverseSets=False):
    """Computes the regularised linear regression
    given a the set of data provided"""
    
    start_time = time.time()
    
    global MaxIterations
    global iterations
    iterations = 0
    lambdaWeight = 0.5
    ftype = 2
    
    print "Linear regression lasso regularised"
    print "Maximum iterations: {0}".format(MaxIterations)
    print "Value of lambda: {0}".format(lambdaWeight)
    print "Regularisation type: {0}".format(ftype)
    
    rawData = read_csv_file(inputFileName) # read the raw data
    expandedValueFeatures = expand_data_features(rawData)
    
    trainingSet, evaluationSet = split_set(expandedValueFeatures, 0.5, reverseSets)
    
    trainingSetValueFeatures = normalise_data(trainingSet) # get actual result-features pair
    theta = initialise_theta(1, trainingSetValueFeatures.featureList.shape[1]) # initial values for theta based
    
    result = fmin_bfgs(lasso_loss, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList, lambdaWeight, ftype),disp=True, full_output=True, maxiter=MaxIterations, callback=monitor_optimization)
    
    evaluationSetHypothesis = evaluationSet.featureList.dot(result[0])
    evaluationSetMeanSquareError = mean_square_error(evaluationSet.value, evaluationSetHypothesis)
    
    m, s = divmod((time.time() - start_time), 60)
    print "Running duration: {0:d}:{1} seconds".format(int(m), s)
    print "Linear regression lasso regularised done\n\n"
    return evaluationSetMeanSquareError

def logistic(inputFileName, reverseSets=True):
    """Computes the logistic regression
    given a the set of data provided"""

    start_time = time.time()
    
    global MaxIterations
    global iterations
    print "Logistic regression"
    print "Maximum iterations: {0}".format(MaxIterations)
    
    rawData = read_csv_file(inputFileName) # read the raw data
    expandedValueFeatures = expand_data_features(rawData, calculate_class)
    trainingSet, evaluationSet = split_set(expandedValueFeatures, 0.5, reverseSets)
        
    trainingSetValueFeatures = normalise_data(trainingSet, True) # get actual result-features pair
    theta = initialise_theta(5, trainingSetValueFeatures.featureList.shape[1]) # initial values for theta based
    
    print "Starting maximisation."
    #theta = fmin_bfgs(logistic_cost, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList),norm=float('+inf'), disp=True, retall=True, maxiter=MaxIterations, callback=monitor_optimization)[0]
    print "Nelder Meading"
    result= minimize(logistic_cost, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList), method="Nelder-Mead", options={'disp':True, 'maxiter':50000 },tol=1e-2, callback=monitor_optimization)
    theta = result['x']
    iterations = 0
    print "BFGSing"
    result= minimize(logistic_cost, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList), method="BFGS", options={'disp':True, 'maxiter':MaxIterations}, callback=monitor_optimization)
    theta = result['x']
    
    theta = theta.reshape(5, trainingSetValueFeatures.featureList.shape[1])
    print "Theta values: "
    print theta
    
    trainingAccuracy = logistic_accuracy(trainingSetValueFeatures, theta)
    #BUG: This will not work
    evaluationAccuracy = logistic_accuracy(evaluationSet, theta)
    
    print LineSeparator
    print "\tTraining Set: Accuracy {0:f}".format(trainingAccuracy)
    print "\tValidation Set: Accuracy {0:f}".format(evaluationAccuracy)
    print "\tDifference: {0:f}".format(abs(evaluationAccuracy - trainingAccuracy))
    m, s = divmod((time.time() - start_time), 60)
    print LineSeparator
    print "Running duration: {0:d}:{1} seconds".format(int(m), s)
    print "Logistic regression done\n\n"
  
def reglogistic(inputFileName, reverseSets=False):
    """Computes the regularised logistic regression
    given a the set of data provided"""

    start_time = time.time()
    
    global MaxIterations
    global iterations
    iterations = 0
    lambdaWeight = 0.3
    
    print "Logistic regression"
    print "Maximum iterations: {0}".format(MaxIterations)
    
    rawData = read_csv_file(inputFileName) # read the raw data
    expandedValueFeatures = expand_data_features(rawData, calculate_class)
    del rawData
    trainingSet, evaluationSet = split_set(expandedValueFeatures, 0.5, reverseSets)
        
    trainingSetValueFeatures = normalise_data(trainingSet, True) # get actual result-features pair
    del trainingSet
    
    theta = initialise_theta(5, trainingSetValueFeatures.featureList.shape[1]) # initial values for theta based
    
    print "Starting maximisation."
    print "Nelder Meading"
    result= minimize(reg_logistic_cost, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList, lambdaWeight), method="Nelder-Mead", options={'disp':True, 'maxiter':  8000},tol=1.0e-2, callback=monitor_optimization)
    m, s = divmod((time.time() - start_time), 60)
    print "Running duration: {0:d}:{1} seconds".format(int(m), s)
    theta = result['x']
    iterations = 0
    print "BFGSing"
    result= minimize(reg_logistic_cost, theta, args=(trainingSetValueFeatures.value, trainingSetValueFeatures.featureList, lambdaWeight), method="BFGS", options={'disp':True, 'maxiter':MaxIterations}, callback=monitor_optimization)
    m, s = divmod((time.time() - start_time), 60)
    print "Running duration: {0:d}:{1} seconds".format(int(m), s)
    theta = result['x']
    
    theta = theta.reshape(5, trainingSetValueFeatures.featureList.shape[1])
    print "Theta values: "
    print theta
    
    trainingAccuracy = logistic_accuracy(trainingSetValueFeatures, theta)
    evaluationAccuracy = logistic_accuracy(evaluationSet, theta)
    
    print LineSeparator
    print "\tTraining Set: Accuracy {0:f}".format(trainingAccuracy)
    print "\tValidation Set: Accuracy {0:f}".format(evaluationAccuracy)
    print "\tDifference: {0:f}".format(abs(evaluationAccuracy - trainingAccuracy))
    m, s = divmod((time.time() - start_time), 60)
    print LineSeparator
    print "Running duration: {0:d}:{1} seconds".format(int(m), s)
    print "Regularised logistic regression done\n\n"

if __name__ == "__main__":
    print(LineSeparator + LineSeparator)
    if (len(sys.argv) > 1):
        print('Processing ...  '  + str(sys.argv) )
        cross_validation(linear, sys.argv[1])
    else:
        print('Input data file not given, using stock_price.csv')
        #cross_validation(reglinear, ".\Data\stock_price.csv")
        logistic(".\Data\stock_price.csv")
    
    