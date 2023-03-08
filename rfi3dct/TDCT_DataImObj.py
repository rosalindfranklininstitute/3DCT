#TODO: Under development


import numpy as np

class TDCT_DataImObj():
    ''' A class that will hold the data from a tiff file
    and has methods and properties that are convenient to use in TDCT_correlation.py
    '''

    def __init__(self, tiffilepath):
        self.datachannels = None
        self.MIP = None
        self.nchannels = 0
        self.bHasSlices = False
        self.slicemin = 0
        self.slicemax = 0

        if not tiffilepath is None:
            self.readtifffile(tiffilepath)
        
    def readtifffile(self, tiffilepath):
        '''
        Reads the data and sets the properties accordingly
        '''
        