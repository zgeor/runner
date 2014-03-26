'''
Common structures and functions used throughout the code.
Created on 15 Feb 2014

@author: Y6187553
'''
from collections import namedtuple

#===============================================================================
# Section with common structures used throughout the code.
#===============================================================================

"""A named tuple to hold data information containing biased xAxis and yAxis.
"""
Data = namedtuple('Data', ['xAxisBiased','yAxis'])

#===============================================================================
# Section with common functions used throughout the code.
#===============================================================================

def read_csv_file(input_filename): #TODO: Change this to be more efficient
    """Read data from a csv file.
    Return the data in a structure
    containing xAxisBiased and yAxis.
    Access them as follows

    Data.xAxisBiased

    Data.yAxis
    """
    data = Data([], [])

    IFile = open(input_filename, 'r')

    for line in IFile:
        line = line.strip()
        row = line.split(',')
        x = [1.0]

        for i in range(len(row)-1):
            cell = float(row[i])
            x.append(cell)

        data.xAxisBiased.append(x)

        data.yAxis.append(float(row[len(row)-1]))

    IFile.close()
    return data
