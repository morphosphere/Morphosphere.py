
# coding: utf-8
#!python2

# Written by Fanny Georgi and Luca Murer for MorphoSphere, 2016

# getFileList.py
# Script that returns absolute paths of all files in an input directory including subfolders, naturally sorted

# Input: inputDirectory as string
# Returns: filenameList as one column numpy string array

# import packages
import os, natsort, numpy
from datetime import datetime

# retrieve basenames of all files in a target directory (including subdirectories)
def getFileList(inputDirectory='.\\'):
    
    # print current time for logging
    print datetime.now()
    print('Reading files from: %s' % inputDirectory)
    
    if not os.path.isdir(inputDirectory):
        print('Path not found.')
        return None
    
    # read basenames of all .TIF files in the input directory (including subdirectories) into array, then sort naturally
    filenameList = []
    i = 0
    tifFiles = re.compile(r'^.*[A-Z][0-9]*_w[0-9]*\.tif$', re.IGNORECASE)
    for foldernames, subfolders, filenames in os.walk(inputDirectory):
        for filename in filenames:
            if tifFiles.match(filename) != None:
                #print(os.path.join(foldernames, filename))
                filenameList.append(os.path.join(foldernames, filename))
    filenameList=natsort.natsorted(filenameList)
    
    # convert file list into 1 colon numpy string array, this makes regexing easier
    filenameList = numpy.array(filenameList, dtype=str).T
    #print filenameList.shape
    #print filenameList
    #print type(filenameList[0])
            
    # display number of files detected
    numberOfFiles = len(filenameList)
    numberOfPlates = numberOfFiles/384
    print "%s files have been detected, equal to %s plates." % (numberOfFiles, numberOfPlates)

    return filenameList

