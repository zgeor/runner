#! /usr/bin/python

# Suresh Manandhar - 16 September 2012 (suresh@cs.york.ac.uk)

# Simple Linear Regression Code (for MLAP module)

# To run enter: ./LinearRegression.py lineData.csv


import os
import sys
import math
import random
import string
import copy
# import bigfloat

LearningRate = 0.0005

TotalLoss = 0.0      # Total 

Data = []            #  is a list of real-valued vectors
Y = []               #  is a sampled values of the function

Grad = []            #  Gradient vector of Theta at current value of Theta
Theta = []           #  the parameter vector -- this is the output of this program

XLen = 0           #  to be set equal to length of Data item/Theta

def read_csv_file(InputFilename):

    global Data, Y

    IFile = open(InputFilename, 'r')
   
    # This part of the code loads input line into X
    
    for line in IFile:
        line = line.strip()
        Row = line.split(',')
        X = [1.0]  # Add bias/constant 1 at X[0]

        for i in range(len(Row)-1):
            Cell = float(Row[i])
            X.append(Cell)

        Data.append(X)
        
        Y.append(float(Row[len(Row)-1])) # Treating last col as Y - change this if different
        
    IFile.close()

    
def initialise_parameters():
    global Theta
    global Grad
    
    # Initialise Parameter, Gradient Table
    for Attr in range(XLen):
        Theta.append(0.0)
        Grad.append(0.0)

def update_parameters():
    global Theta
    
    for Attr in range(XLen):
        Theta[Attr] += (LearningRate * Grad[Attr])

def dot_product(V1, V2): # Implemented
    Sum = 0.0

    for i in  range(len(V1)):
        Sum += V1[i]*V2[i]

    return(Sum)


def compute_loss_and_gradient():

    global Theta, Grad, TotalLoss

    TotalLoss = 0.0

    # Initialise Gradient Table
    for Attr in range(XLen):
        Grad[Attr] = 0.0

    for i in  range(len(Data)):
        X =  Data[i]
        y_computed = dot_product(X, Theta)
        # Update gradient of each parameter
        Error = float(Y[i] - y_computed)
        for Attr in range(XLen):
            Grad[Attr] += Error * X[Attr]

        TotalLoss += Error**2


def main(InputFilename):
    global LearningRate, XLen
    
    read_csv_file(InputFilename)
    
    XLen = len(Data[0]) # initialise vector length using the first data item
    
    initialise_parameters()

    OFile = open('output.csv', 'w')

    i = 0
    for i in range(100000):
        compute_loss_and_gradient()
        # OFile.write(str(Theta[0]) + str(',') + str(Theta[1]) + str(',') + str(TotalLoss) +  str('\n'))
        # print(str(Theta[0]) + str(',') + str(Theta[1]) + str(',') + str(TotalLoss) +  str('\n'))
        update_parameters()
        if (TotalLoss < 4):
            break


    OFile.write(str(Theta[0]) + str(',') + str(Theta[1]) + str(',') + str(TotalLoss) +  str('\n'))
    OFile.close
    print("Total Loss = " + str(TotalLoss))
    print("Total Iterations  =  " + str(i))
    print('Parameter values ')
    print(Theta)


if __name__ == "__main__":
    if (len(sys.argv) > 1):
        print('Processing ...  '  + str(sys.argv) )
        main(sys.argv[1])
    else:
        print('Input data file not given, using ..\stock_price.csv')
        main("..\stock_price.csv")