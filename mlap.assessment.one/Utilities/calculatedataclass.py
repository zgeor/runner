'''
Created on 13 Feb 2014

@author: Y6187553
'''
import csv

def calculate_class(file_name):
    last_row_value = 0
    with open(file_name, 'w+') as file: 
        reader = csv.reader(file)
        for row in reader:
            print ', '.join(row)

calculate_class('stock_price.csv')