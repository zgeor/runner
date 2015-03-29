#!/usr/bin/python
'''
Created on 15 Mar 2015

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
import json

from mpl_toolkits.mplot3d import axes3d
from scipy import integrate, interpolate
from collections import namedtuple
from math import sin 
from operator import itemgetter
from matplotlib.pylab import *
from pandas.core.frame import DataFrame

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


def evaluateModel(model, data):
    """
    Evaluates a model given all datapoints required.
    
    string     model         the model to be evaluated
    dict or recarray       variables     the variables of the model
    
    Remark    Need to restrict the global values used for evaluation.
    
    return float
    """
    return eval(model.replace('variable_', ''), globals(), data)
    #return DataFrame.from_records(data).eval(model.replace('variable_', ''))

def getModel(eqs, modelName):
    
    for equation in equations:
        if equation['variable'] == 'og':
           return equation['equation']
        if equation['variable'] == 'ri':
            return equation['equation']
        if equation['variable'] == 'in':
            return equation['equation']
         
    return
def generatePhasePlotExample(eqs):
    
    # Generate 20 evenly spaced numbers from -2 until 8
    y1 = numpy.linspace(-2.0, 8.0, 20)
    y2 = numpy.linspace(-2.0, 2.0, 20)
    
    # The sum of Y1 and Y2 is a matrix of the combination of the numbers in the vectors
    Y1, Y2 = numpy.meshgrid(y1, y2)
    
    data = dict()
    
    data['y1'] = Y1
    data['y2'] = Y2
            # calculate a derivative at this point
    u1 = evaluateModel("y2", data)
    v1 = evaluateModel("-numpy.sin(y1)", data)
    
    Q = plt.quiver(Y1, Y2, u1, v1, color='r')
    
    plt.xlabel('$y_1$')
    plt.ylabel('$y_2$')
    plt.xlim([-2, 8])
    plt.ylim([-4, 4])
    #plt.savefig(outputFileName)
    plt.show()
    
def plotSin():
    x = numpy.linspace(0, 10)
    y = numpy.sin(x)
    plt.plot(x, y)
    plt.xlabel('t')
    plt.ylabel('y')
    plt.ylim([-1.5, 1.5])
    plt.grid(b=True, which='major', color='r', linestyle='-')
    plt.show()
    
def plotMonod():
    x = numpy.linspace(0, 6)
    t = numpy.linspace(0, 4)
    y = monod(x, t)
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('monod')
    plt.grid(b=True, which='major', color='r', linestyle='-')
    plt.show()
    
def monod(c, v): 
    return (v / (v + c));

def generatePhasePlot(eqs):
    # Generate 20 evenly spaced numbers from -2 until 8
    y1 = numpy.linspace(-2.0, 8.0, 20)
    y2 = numpy.linspace(-2.0, 2.0, 20)
    
    data = dict()
    maxPoints = 5
    endPoint = 4
     
    og = numpy.linspace(0.0, 5, maxPoints)
    infl = numpy.linspace(0.0, 3, maxPoints)
    ri = numpy.linspace(0.0, 2, maxPoints)
     
    
    # The sum of Y1 and Y2 is a matrix of the combination of the numbers in the vectors
    data['og'], data['infl'], data['ri'] = numpy.meshgrid(og, infl, ri)
    
    
#     u, v = numpy.zeros(data['og'].shape), numpy.zeros(data['infl'].shape)
#      
#     NI, NJ = data['og'].shape
#  
#     for i in range(NI):
#         for j in range(NJ):
#             x = Y1[i, j]
#             y = Y2[i, j]
#              
#             yprime = f([x, y], t)
#              
#             u[i,j] = yprime[0]
#             v[i,j] = yprime[1]
         
     # calculate a derivative at this point
    u = evaluateModel(eqs[0]['equation'], data)
    w = evaluateModel(eqs[1]['equation'], data)
    v = evaluateModel(eqs[2]['equation'], data)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.quiver(data['og'], data['infl'], data['ri'], u, v, w, length=0.1)
    
#     plt.xlabel('og')
#     plt.ylabel('infl')
    #plt.savefig(outputFileName)
    plt.show()
    
def generateMatrixData(variables):
    y1 = numpy.linspace(-2.0, 8.0, 20)
    
    data = numpy.recarray((y1.size,), names=variables, formats=['f8', 'f8','f8'])
    for i in variables:
        data[i] = y1
    return data

def loadJsonData(fileName):
    """
    Parses JSON file.
    
    Returns a dictionary.
    """
    with open(fileName) as json_data:
        data = json.load(json_data)
    return data

def parseProgramArgs(args):
    """
    Parses the input arguments of the program.
    Overwrites the local configuration file.
    
    returns dict<dict>
    """

    parser = argparse.ArgumentParser(description='Run plotter equation plotting tool.')
    
    parser.add_argument('equationsFile', metavar='Json file with the system of equations.', type=str,
                       help='The equations to be plotted.')
#     parser.add_argument('-e', dest='eqs', required=True, type=str, help='Lagramge equations json file.')
    parser.add_argument('-o', dest='outFile', required=True, type=str, help='Output file for the plot.')
#     parser.add_argument('-t', dest='timeStep', default=False, type=float32, help='Time step for differential equations. By default ')

    parsed = parser.parse_known_args()

    return parsed[0].equationsFile, parsed[0].outFile

def main(argv):
    """
    Main method that unfortunately is way too complicated.
    """    

    eqsFile, plotFile = parseProgramArgs(argv)
    plotMonod()
#     equations = loadJsonData(eqsFile)['equations']
#     generatePhasePlot(equations)

    #data = generateMatrixData(("og", "in", "ri"))
#     data = dict()
#     maxPoints = 40
#     endPoint = 8
#     maxPoints2 = 40
#     endPoint2 = -2.0
#     space = numpy.linspace(0.0, endPoint, maxPoints)
#     spaceInit = space[:space.size - 2]
#     spaceLagged1 = space[1:space.size - 1]
#     spaceLagged2 = space[2:space.size]
#     
#     space2 = numpy.linspace(0.0, endPoint2, maxPoints2)
#     spaceInit2 = space[:space2.size - 2]
#     spaceLagged12 = space[1:space2.size - 1]
#     spaceLagged22 = space[2:space2.size]
#     
#     data['og'] = spaceInit
#     data['infl'] = spaceInit2
#     data['ri'] = spaceInit
#     
#     data['rima'] =  spaceLagged1
#     data['rimb'] =  spaceLagged2
#     data['inflma'] =  spaceLagged12
#     data['inflmb'] =  spaceLagged22
#     data['ogma'] =  spaceLagged1
#     data['ogmb'] =  spaceLagged2
#     
#     x, y = np.meshgrid(space, space2)
#     
#     u = None
#     v = None
#     w = None
#     for equation in equations:
#         if equation['variable'] == 'og':
#             #u = evaluateModel(equation['equation'], data)
#             u = evaluateModel("-numpy.sin(og)", data)
#         if equation['variable'] == 'ri':
#             w = evaluateModel(equation['equation'], data)
#         if equation['variable'] == 'in':
#             #v = evaluateModel(equation['equation'], data)
#             v = evaluateModel("infl", data)
#     
#     q = plt.quiver(x, y, u, v, edgecolor='k', alpha=.5)
#     plt.xlabel('og')
#     plt.ylabel('infl')
#     plt.show()
    
if __name__ == '__main__':
    main(sys.argv[1:])