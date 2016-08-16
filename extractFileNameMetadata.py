
# coding: utf-8
#!python2

# Written by Fanny Georgi, Luca Murer and Yardan Andriasyan for MorphoSphere, 2016

# This script reads information from file names from a one column numpy string array according to 
# regular expressions. Note that these need to be custom-adjusted for specific file architectures and filenames.

# Input: fileList as one column numpy string array
# Returns: filenameMetadata as numberOfSelectedFiles x keys numpy character array
         # selectedFilenameList as one column numpy string array

# import packages
import os, re, numpy

# extract filename metadata into numpy character array
def extractFileNameMetadata(fileList):
    
    filenameMetadata = numpy.chararray((numberOfFiles,7), itemsize=200)

    # expected format A is 'N:\\FannyGeorgi\\1-9_Virusprep\\1-9-4_Virusprep_HAdV-C5_dl312_Alexa488\\20160705-HAdV-dl312-pre-post-1dpi_Plate_2917\\Timepoint_1\\20160126-corning-all-spheroids-p1-003hps_A01_w1.TIF'
    # define the regex pattern A
    #filePattern = re.compile(r'.+\\Timepoint_(?P<timePoint>\d+)\\(?P<date>\d+)-(?P<experimentName>.*)-p(?P<plate>\d+)-(?P<hps>\d+)hps_(?P<wellID>\w\d\d)_w(?P<channel>\d+).TIF', re.IGNORECASE)

    # expected format B is 'C:\\Users\\FannyG\\Documents\\Data\\20151217_1-12_TumorRemission\\MorphoSphereTest\\1\\0dps\\20160126-corning-all-spheroids-p1-003hps_A01_w1.TIF'
    # define regex pattern B
    filePattern = re.compile(r'.+\\(?P<date>\d+)-(?P<experimentName>.*)-p(?P<plate>\d+)-(?P<hps>\d+)hps_(?P<wellID>\w\d\d)_w(?P<channel>\d+).TIF', re.IGNORECASE)

    selectedFilenameList = []
    
    # define wells, channels etc to be analyzed to filter list of files
    acceptedWells = []
    acceptedRows = ['A', 'B']
    acceptedColums = ['01', '02']
    for r in acceptedRows:
        for c in acceptedColums:
            acceptedWells.append(r+c)   
        
    acceptedChannels = ['1', '2']

    i = 0
    j = 0

    for i in fileList:
        match = filePattern.search(i)
        if match.group('channel') in acceptedChannels and match.group('wellID') in acceptedWells:
            filenameMetadata[j, 0] = i
            filenameMetadata[j, 1] = match.group('date')
            filenameMetadata[j, 2] = match.group('experimentName')
            filenameMetadata[j, 3] = match.group('plate')
            filenameMetadata[j, 4] = match.group('hps')
            filenameMetadata[j, 5] = match.group('wellID')
            filenameMetadata[j, 6] = match.group('channel')
            selectedFilenameList.append(filenameMetadata[j, 0])
            j = j+1 
            
        #!!! temporal solution due to memory leak, needs optimization
        else:
            filenameMetadata[j, 0] = 0
            filenameMetadata[j, 1] = 0
            filenameMetadata[j, 2] = 0
            filenameMetadata[j, 3] = 0
            filenameMetadata[j, 4] = 0
            filenameMetadata[j, 5] = 0
            filenameMetadata[j, 6] = 0
            j = j+1 

    # remove empty rows using logical indexing
    filenameMetadata = filenameMetadata[~(filenameMetadata[:,0] == '0')]   
    #print filenameMetadata   
    #print len(filenameMetadata.T)
    #print filenameMetadata.shape

    selectedFilenameList = numpy.array(selectedFilenameList, dtype=str).T
    #print selectedFilenameList

    return filenameMetadata, selectedFilenameList
