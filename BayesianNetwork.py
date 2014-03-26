'''
Created on 27 Mar 2014

@author: Y6187553
'''
#===============================================================================
# Section with imported modules. 
# Numpy functions will be imported one by one
# and will always be prepended by 'np' for easier tracking.
#===============================================================================
import sys
from numpy import loadtxt as nploadtxt, where as npwhere, empty as npempty, nan as npnan, in1d
from random import random
from itertools import product, izip
from collections import namedtuple, deque

#===============================================================================
# Section with custom structures.
#===============================================================================
PosteriorInfo = namedtuple('PosteriorInfo', '')
#===============================================================================
# Section with functions used in the code.
#===============================================================================
def read_csv(filePath):
    """
    Reads a csv file into a matrix.
    
    filePath - the path to the file
    
    NOTE: broken csv's will throw an exception.
    """
    data = nploadtxt(filePath, delimiter=',')
    return data

def get_ones_indexes_col(structureMatrix, column):
    """
    Gets the indexes of the nodes in the column(index) where value is 1
    """
    structCol = structureMatrix[:,column]
    return structCol.nonzero()[0];

def get_zeros_indexes_col(structureMatrix, column):
    """
    Gets the indexes of the nodes in the column(index) where value is 1
    """
    structCol = structureMatrix[:,column]
    return npwhere(structCol == 0)[0];

def generate_binary_permutations(length):
    """
    Generates binary permutations long as the length specified.
    """
    arr = product([0, 1], repeat=length)
    return list(arr)

def count_rows(dataMatrix, columns, perm):
    dataMatrix = dataMatrix[:, columns]
    count = 0
    for i in range(dataMatrix.shape[0]):
        if(tuple(dataMatrix[i]) == perm):
            count+=1;
    return count;

#===============================================================================
# Section with global variables used throughout the code.
#===============================================================================

#===============================================================================
# Section with main functions required by the assessment.
#===============================================================================
def bnbayesfit(StructureFileName, DataFileName):
    """
    Estimates the parameters of a Bayesian network
    given its DAG structure and data.
    
    A uniform prior is put on each parameter.
    Each parameter is estimated independently .
    """
    structureMatrix = read_csv(StructureFileName)
    dataMatrix = read_csv(DataFileName)
    
    # i is the variable index
    # permutations[j] is a binary permutation of parent variables
    # post is the posterior
    # out returns only probabilities the variable to be equal to 1 given the permutation
    # the negative probability is acquired by subtracting the probability of 1 from 1
    out = []
    for i in range(structureMatrix.shape[0]):
        dependentOn = get_ones_indexes_col(structureMatrix, i)
        out.append((dependentOn, []))
        # Using permutations for readability, could be encoded with integers only
        permutations = generate_binary_permutations(dependentOn.shape[0])
        
        dataMatrixOnesIndexes = get_ones_indexes_col(dataMatrix, i)
        dataMatrixZerosIndexes = get_zeros_indexes_col(dataMatrix, i)
        
        dataMatrixOnes = dataMatrix[dataMatrixOnesIndexes,]
        dataMatrixZeros = dataMatrix[dataMatrixZerosIndexes,]
        if(len(permutations) == 1):
            post = (1.0 + len(dataMatrixOnesIndexes))/(2.0 + len(dataMatrixOnesIndexes) + len(dataMatrixZerosIndexes))
            out[i][1].append((None ,post))
        else:
            for j in range(len(permutations)):
                countVarPositive = count_rows(dataMatrixOnes, dependentOn, permutations[j])
                countVarNegative = count_rows(dataMatrixZeros, dependentOn, permutations[j])
                post = (1.0 + countVarPositive)/(2.0 + countVarPositive + countVarNegative)
                out[i][1].append((permutations[j], post))

    return out

def bnsample(fittedbn,nsamples):
    variablesCount = len(fittedbn)
    dependencies = resolve_dependencies(fittedbn)
    dependencies = order_dependencies(dependencies)
    samples = npempty(shape=(nsamples, variablesCount), dtype=int)
    for i in range(nsamples):
        sample = npempty(variablesCount, dtype=int)
        sample.fill(npnan)
        for j in dependencies:
            parents = resolve_parent_value_tuple(sample, fittedbn[j][0])
            randomSample = random()
            positiveTreshold = -1
            if(parents == None):
                positiveTreshold = fittedbn[j][1][0][1]
            else:
                for perm in fittedbn[j][1]:
                    if in1d(parents, perm[0]).all():
                        positiveTreshold = perm[1]
                        break
            if(positiveTreshold >= randomSample):
                sample[j] = 1
            else:
                sample[j] = 0               

        samples[i] = sample
    return samples

def resolve_parent_value_tuple(sample, parents):
    parents = [sample[n] for n in parents]
    if(len(parents) == 0):
        return None
    else:
        return tuple(parents)
    
def order_dependencies(dependencies):
    unsorted = deque(enumerate(dependencies))
    sorted = []
    while unsorted:
        dep = unsorted.popleft()
        if(len(dep[1]) == 0):
            sorted.append(dep[0])
        else:
            if(in1d(dep[1], sorted).all()):
                sorted.append(dep[0])
            else:
                unsorted.append(dep)
    return sorted

def resolve_dependencies(fittedbn):
    lenFitted = len(fittedbn)
    dependencies = []
    for i in range(lenFitted):
        dependencies.append(fittedbn[i][0])
    return dependencies
    
#===============================================================================
# Main method for executing code.
#===============================================================================
if __name__ == '__main__':
        print('Processing ... files in Data\ folder. '  + str(sys.argv) )
        fittedBn = bnbayesfit("Data\\bnstruct.bn", "Data\\bndata.csv")
        bnsample(fittedBn, 100)

