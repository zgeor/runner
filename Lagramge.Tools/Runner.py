#!/usr/bin/python
'''
Created on 19 Dec 2014

@author: Zhivko Georgiev
'''

#####################################
# User imports
#####################################
import sys
import signal
import os.path
import json
import argparse
from subprocess import Popen
import string
import time
import re
import numpy as numpy
from scipy import integrate, interpolate
from collections import namedtuple
import csv
import uuid
from math import sin 
from os.path import basename
from operator import itemgetter
from matplotlib.pylab import *
import warnings
import glob
#####################################
# Setup variables
#####################################
TempDataFolder = "temp/"
LagramgeMainExe = "./g"
ValidCharsFileName = "-_()%s%s" % (string.ascii_letters, string.digits)

#####################################
# Global variables
#####################################
OutputFiles = []
Processes = None
Configuration = dict()
ModelParsingRegExpGroup = r'.* = (.*){MSE = ((\d*\.\d*)|inf|-inf), MDL = ((\d*\.\d*)|inf|-inf)}$'
ModelMeasurmentMatcher = re.compile(ModelParsingRegExpGroup)

#####################################
# Custom structures
#####################################
LModel = namedtuple("LModel", "Eq Mse Mdl")

#####################################
# Functional code
#####################################
def parseLagramgeOutput(fileName):
    """
    This parses a Lagramge output file(since there seems to be a bug with -o option
    in the program, this will take output as seen on the screen(i.e. print).
    Parses the output and takes all the models and assembles LModel struct. 
    This struct provides MSE and MDL values as well as the model itself.
    
    returns List<LModel>
    """
    
    with open(fileName) as output:
        content = output.readlines()
    indexOfFirstEq = -1
    lastIndexOfEq = -1
    contentLength = len(content)
    for x in range(0, contentLength):
        if(content[x].startswith("Best equations:")):
            indexOfFirstEq = x + 1
            break
            
    for x in range(contentLength - 1, indexOfFirstEq, -1):
        if(content[x].startswith("Time elapsed:")):
            lastIndexOfEq = x - 2
            break
        
    if(indexOfFirstEq < 1):
        raise Exception("Could not find beginning of model section.")
    
    if(lastIndexOfEq < indexOfFirstEq):
        raise Exception("Could not find any models.")
    models = []
    for x in range(indexOfFirstEq, lastIndexOfEq + 1):
        search = ModelMeasurmentMatcher.search(content[x])
        group = search.group(1, 2, 3)
        models.append(LModel(group[0], group[1], group[2]))
    return models

def readData(fileName):
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

def prepDataRow(row, dataLists):
    """
    Gets dictionary of column names and row values.
    
    returns dict<str, float>, int
    """
    return dict(zip(dataLists[0], dataLists[row]))

def splitDataForCV(dataFile, foldSizes):
    """
    Splits the input data file into folds specified in percentages.
    
    returns List<str>
    """
    with open(dataFile) as f:
        content = f.readlines()
        
    namePattern = 'fold_{0}.data'
    args = foldSizes
    
    sumOfFolds = sum(foldSizes)
    if(sumOfFolds > 1):
        raise Exception("Invalid fold sizes.")
    
    if(sumOfFolds < 1):
        args.append(1 - sumOfFolds)
    
    folds = []
    folds.append((1, int(round((len(content) - 1) * args[0]))))
    
    for x in range(1, len(args)):
        folds.append((folds[x-1][1], folds[x-1][1] + int(round((len(content)) * args[x]))))
    
    lastFold = folds.pop()
    if(lastFold[1] != len(content)):
        folds.append((lastFold[0], len(content)))
    else:
        folds.append(lastFold)
        
    dataFiles = []
    for y in range(0, len(folds)):
        fileName = TempDataFolder + namePattern.format(y)
        dataFiles.append(fileName)
        with open(fileName, 'w') as fold:
            fold.write("%s" % content[0])
            for z in range(folds[y][0], folds[y][1]):
                if content[z]:
                    fold.write("%s" % content[z])
    return dataFiles

def preProcessData(fileName, processingFunction):
    """
    This is data pre-processing step.
    
    """
    dataWithHeader, length = readData(fileName)
    header = dataWithHeader[0]
    data = numpy.array(dataWithHeader[1:])

def minMaxNormalisation(data):
    nprollaxis(data, axis=0)

    mean = data.mean(axis=0)
    std = data.std(axis=0)
    if isinstance(std, ndarray):
        std[std == 0.0] = 1.0
    elif std == 0.:
        std = 1.
    data -= mean
    data /= std
        
def generateLagramgeCommand(dataFileName, localConfig):
    """
    Generate Lagramge command line arguments.
    
    return str
    """
    cmdLine = LagramgeMainExe
    for (key, value) in localConfig.items():
        if(value != None):
            if(value == True):
                cmdLine += ' ' + key
                break
            if(value != False):
                cmdLine += ' ' + key + ' ' + str(value)
    cmdLine += ' ' + dataFileName
    #fileName = ''.join(c for c in cmdLine if c in ValidCharsFileName)
    print "CMD: %s" % cmdLine
    return cmdLine

def parseConfigFile(configFileName):
    """
    Parses the JSON configuration file of the runner.
    
    Returns a dictionary.
    """
    with open(configFileName) as json_data:
        data = json.load(json_data)
    return data

def stillRunning(processes):
    """
    Checks if processes are still running and prints elapsed time info.
    Returns true if any process is still running.
    """
    printout = ''
    running = False
    for p in processes:
        if(p.poll() == None):
            running = True
            printout += 'PID: ' + str(p.pid) + ' is running. '
        else:
            printout += 'PID: ' + str(p.pid) + ' is ' + str(p.poll()) + '. '
    print time.strftime("%d %b %Y %H:%M:%S ", time.gmtime()) + printout + '\r'
    return running

def addOutputFile(fileName):
    """
    Create the file name and add it to global list.
    
    return str
    """
    global OutputFiles
    fileName = Configuration['runner']['outputFolder'] + fileName + '.log'
    OutputFiles.append(fileName)
    return fileName

def addOutputFileById(runId):
    """
    Create the file name with id and add it to global list.
    
    return str
    """
    global OutputFiles
    fileName = Configuration['runner']['outputFolder'] + runId + '.log'
    print "OutputFile: ", fileName
    OutputFiles.append(fileName)
    return fileName

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

def forwardEulerIntegration(calculated, actual, timeStep):
    """
    Implementation of forward Euler integration method.
    """
    i = 0
    output = numpy.zeros((actual.size, ))
    
    summation = output[0] = actual[0]
    
    for i in range(1, actual.size):
        summation += (calculated[i -1])* timeStep
        output[i] = summation 
    return output

def AdamBashforth2Integration(calculated, actual, timeStep):
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

def AdamBashforth2Corrector(predicted, calculated, actual, timeStep):
    """
    Implementation of Adam-Bashforth 2 correctorMethod.
    """
    output = numpy.zeros((actual.size, ))
    summation = output[0] = actual[0]
    
    for i in range(1, actual.size):
        summation += (calculated[i] - calculated[i-1])*(1/2)* timeStep
        output[i] = summation 
    return output

def rateModels(lOutputFileName, dataFileName):
    """
    Rates the Lagramge models, according to some Accuracy results.
    
    returns dict<str, dict<str, str>>
    """
    global Configuration
    results = dict()
    # "D:\\Lagramge\\downloads\\results\\OG-gstep.7.gramm-hmse-sexhaustive-d5-hmse.log"
    models = parseLagramgeOutput(lOutputFileName)
    
    # "D:\\Lagramge\\downloads\\temp\\trainDataOGnRI1.csv"
    preppedData, dataLength = readData(dataFileName)
    results['isValidation'] = True
    results['dataLength'] = dataLength
    results['isDifferential'] = bool(Configuration['lagramge']['-i'] or Configuration['lagramge']['-t'])
    timeStep = 1
    
    if(results['isDifferential'] and Configuration['lagramge']['-i']):
        timeStep = Configuration['lagramge']['-i']
    
    results['models'] = dict()
    
    for i, model in enumerate(models):
        results['models'][i] = dict()
        results['models'][i]['equation'] = model.Eq
        results['models'][i]['lagramgeMSE'] = model.Mse
        results['models'][i]['lagramgeMDL'] = model.Mdl
        results['models'][i]['runMSE'] = 0.0
        results['models'][i]['runMPE'] = 0.0
        results['models'][i]['runMAPE'] = 0.0
            
    pVarName = Configuration['lagramge']['-v']

    if results['isDifferential']:
        for i in results['models']:
            evaluationDataPoints = 0
            calculated = numpy.zeros((dataLength - 1, ))
            
            for data in preparedDataRow(preppedData):
                calculated[evaluationDataPoints] = evaluateModel(results['models'][i]['equation'], data)
                evaluationDataPoints += 1
                
            actual = numpy.array(map(itemgetter(preppedData[0].index(pVarName)), preppedData[1:dataLength]))
            predicted = AdamBashforth2Integration(calculated, actual, timeStep)
            
            evaluationDataPoints = 0
            corrected = numpy.zeros((dataLength - 1, ))
            for data in preparedDataRow(preppedData):
                data[pVarName] = predicted[evaluationDataPoints]
                corrected[evaluationDataPoints] = evaluateModel(results['models'][i]['equation'], data)
                evaluationDataPoints += 1
                
            error = numpy.subtract(actual, corrected)
            squaredError = numpy.multiply(error, error)
            mpe = numpy.average(numpy.divide(error, actual)) * 100.0
            mape =  numpy.average(numpy.abs(numpy.divide(error, actual))) * 100.0
            mse = numpy.average(squaredError)
            
            results['models'][i]['runMSE'] =  mse
            results['models'][i]['runMPE'] = mpe
            results['models'][i]['runMAPE'] =  mape
    else:
        evaluationDataPoints = 0.0
        for data in preparedDataRow(preppedData):
            evaluationDataPoints += 1
            for i in results['models']:
                res = evaluateModel(results['models'][i]['equation'], data)
                results['models'][i]['runMSE'] += calcSquaredError(data[pVarName], res)
                results['models'][i]['runMPE'] += calcPercentageError(data[pVarName], res)
                results['models'][i]['runMAPE'] += calcAbsolutePercentageError(data[pVarName], res)
                
        for i in results['models']:
            results['models'][i]['runMSE'] = results['models'][i]['runMSE']/evaluationDataPoints
            results['models'][i]['runMPE'] = results['models'][i]['runMPE']/evaluationDataPoints
            results['models'][i]['runMPE'] = results['models'][i]['runMAPE']/evaluationDataPoints
        
    results['bestMseMId'] = getBestModel(results['models'], "runMSE")
    results['bestMpeMId'] = getBestModel(results['models'], "runMPE")
    results['bestMapeMId'] = getBestModel(results['models'], "runMAPE")
    results['bestMse'] = results['models'][results['bestMseMId']]['runMSE']
    results['bestMape'] = results['models'][results['bestMpeMId']]['runMAPE']
    results['bestMpe'] = results['models'][results['bestMpeMId']]['runMPE']
    return results

def getBestModel(results, instrument):
    return min(results, key=lambda v: results[v].get(instrument))

def getBestMpeModel(results, instrument):
    return min(results, key=lambda v: abs(results[v].get(instrument)))

def calcSquaredError(actualResult, forecastResult):
    """
    Calculates Squared error.
    
    returns float
    """
    res = 0.0
    try:
        res = (actualResult - forecastResult)**2
    except Warning:
        print actualResult
        print forecastResult
    return res

def calcPercentageError(actualResult, forecastResult):
    """
    Calculates Percentage error 
    
    returns float
    """
    return ((actualResult - forecastResult)/actualResult) * 100

def calcAbsolutePercentageError(actualResult, forecastResult):
    """
    Calculates Absolute Percentage error.
    
    returns float
    """
    return (abs((actualResult - forecastResult)/actualResult)) * 100

def setupDirectories(uniqueRunId):
    global Configuration, TempDataFolder
    try:
        os.mkdir(TempDataFolder)
    except Exception, e:
        print "Warning: Directory %s might exist %s" % (TempDataFolder, str(e))
    
    TempDataFolder += uniqueRunId + '/'
    try:
        os.mkdir(TempDataFolder)
    except Exception, e:
        print "Warning: Directory %s might exist %s" % (TempDataFolder, str(e))
    try:
        os.mkdir(Configuration['runner']['errorFolder'])
    except Exception, e:
        print "Warning: Directory %s might exist %s" % (Configuration['runner']['errorFolder'], str(e))
    try:
        os.mkdir(Configuration['runner']['outputFolder'])
    except Exception, e:
        print "Warning: Directory %s might exist %s" % (Configuration['runner']['outputFolder'], str(e))
    try:
        os.mkdir(Configuration['runner']['jsonFolder'])
    except Exception, e:
        print "Warning: Directory %s might exist %s" % (Configuration['runner']['outputFolder'], str(e))

def signalHandler(signal, frame):
    """
    Send kill signal to lagramge processes.
    
    return void
    """
    global Processes
    print('Stopping Lagramge!')
    for proc in Processes:
        proc.send_signal(signal)

def writeJsonToFile(fileName, jsonObject):
    f = open(fileName,'w')
    f.write(json.dumps(jsonObject, indent=3)) 
    f.close()
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

#####################################
# Program execution functions
#####################################

def parseProgramArgs(args):
    """
    Parses the input arguments of the program.
    Overwrites the local configuration file.
    
    returns dict<dict>
    """

    parser = argparse.ArgumentParser(description='Run LAGRAMGE cross-validation process.')
    
    parser.add_argument('conf', metavar='runc file', type=str, nargs='?', default="runc/default.runc",
                       help='The runner configuration file to be used. If none then default.runc will be used.')
    
    parser.add_argument('-r', dest='reevaluate', default=False, action="store_true", help='Re-evaluate results.')
    parser.add_argument('-d', dest='diffsOnly', default=False, action="store_true", help='Re-evaluate only differential equations.')
    
    parsed = parser.parse_known_args()

    return parsed[0].conf, parsed[0].reevaluate, parsed[0].diffsOnly

def main(argv):
    """
    Main method that unfortunately is way too complicated.
    """    
    #warnings.filterwarnings('error')
    global Configuration, Processes

    confName, reeval, diffsOnly = parseProgramArgs(argv)

#    Configuration = parseConfigFile(confName)
#    Configuration['name'] = confName
    
#     results = rateModels("D:\\Lagramge\\downloads\\08943fde-aef6-11e4-b51a-00155d83ed12.log", "D:\\Lagramge\\downloads\\all_var_train_l.csv")
#     results['configuration'] = Configuration
#     f = open("C:\\inetpub\\wwwroot\\view\\lres\\derr-test.json",'w')
#     f.write(json.dumps(results, indent=3)) 
#     f.close() 
    if(not reeval):
        runId = str(uuid.uuid1())
         
        Configuration = parseConfigFile(confName)
        Configuration['name'] = confName
         
        # Setup interruption signal for graceful exit of lagramge.
        signal.signal(signal.SIGINT, signalHandler)
         
        setupDirectories(runId)
         
        dataFiles = splitDataForCV(Configuration['runner']['inputDataFile'], Configuration['runner']['folds'])
        validationSet = dataFiles.pop()
        commands = []
        for dataFile in dataFiles:
            commands.append(generateLagramgeCommand(dataFile, Configuration['lagramge']))
               
        Processes = [Popen(cmd, 
                           stdout=open(addOutputFileById(runId), 'w'),
                           stderr=open(Configuration['runner']['errorFolder'] + runId + '.log', 'w'), shell=True) for cmd in commands]
         
        while stillRunning(Processes):
            time.sleep(10)
             
        results = rateModels(OutputFiles[0], validationSet)
        results['configuration'] = Configuration
         
        writeJsonToFile(Configuration['runner']['jsonFolder'] + basename(runId) + '.json', results) 
    else:
        for jFile in glob.glob(confName + "*.json"):
            isDifferential = False
            with open(jFile) as jsonFile:
                initResults = json.load(jsonFile)
                isDifferential = initResults['isDifferential']
                Configuration = initResults['configuration']
            if(not(diffsOnly and not isDifferential)):
                fileBase = os.path.basename(jFile)
                runId = string.split(fileBase, ".")[0]               
                 
                results = rateModels(Configuration['runner']['outputFolder'] + runId + '.log', TempDataFolder + runId + "/fold_1.data")
                results['configuration'] = Configuration
     
                writeJsonToFile(Configuration['runner']['jsonFolder'] + basename(runId) + '.json', results)
     
    print "KOHEC"

if __name__ == '__main__':
    main(sys.argv[1:])