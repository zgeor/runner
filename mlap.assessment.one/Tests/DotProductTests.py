'''
Created on 15 Feb 2014

@author: Y6187553
'''
import unittest
from mainSubmission import dot_product

class DotProductTests(unittest.TestCase):

    def test_not_equal_length_vectors_exception_thrown(self):
        self.assertRaises(ValueError, dot_product, [1, 2, 3], [2, 1])
        
    def test_equal_length_vectors_no_exception(self):
        self.assertIsNotNone(dot_product([1, 2, 3], [2, 1, 3]), "No value returned from the dot_product.")
        
    def test_accurate_sum_returned(self):
        self.assertEqual(20, dot_product([1, 2, 3], [2, 3, 4]), "Unexpected dot product result.")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()