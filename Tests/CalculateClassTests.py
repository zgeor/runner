'''
Created on 22 Mar 2014

@author: Y6187553
'''
import unittest
from mainSubmission import calculate_class

class CaclulateClassTests(unittest.TestCase):


    def calculateClassThree_Success(self):
        self.assertEqual(3, calculate_class(9, 4.5), "Unexpected class calculation result.")
    
    def calculateClassFour_Success(self):
        self.assertEqual(4, calculate_class(4.5, 9), "Unexpected class calculation result.")
    
    def calculateClassZero_Success(self):
        self.assertEqual(1, calculate_class(4.5, 4.6), "Unexpected class calculation result.")
        
    def calculateClassTwo_Success(self):
        self.assertEqual(2, calculate_class(9, 9.8), "Unexpected class calculation result.")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()