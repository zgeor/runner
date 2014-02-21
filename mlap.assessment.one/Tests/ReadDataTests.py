'''
Created on 15 Feb 2014

@author: Y6187553
'''
import unittest
import TestSettings
from mainSubmission  import read_csv_file

class ReadDataTests(unittest.TestCase):

    def test_price_correct_values(self):
        data = read_csv_file(TestSettings.CsvTestFile)
        for i in range(0, 10):
            self.failUnless(data.price[i] == i)

    def test_volume_correct_data_values(self):
        data = read_csv_file(TestSettings.CsvTestFile)
        for i in range(0, 10):
            self.failUnless(data.volume[i] == i)
    
            
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()