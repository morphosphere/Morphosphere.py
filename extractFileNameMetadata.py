
# coding: utf-8
#!python2

# Written by Fanny Georgi and Luca Murer for MorphoSphere, 2016

# This script reads information from file names from a one column numpy string array according to 
# regular expressions. Note that these need to be custom-adjusted for specific file architectures and filenames.

# Input: inputDirectory as one column numpy string array
# Returns: filenameMetadata as numberOfFiles x keys numpy character array

# import packages
import os, re, numpy

# extract filename metadata into numpy character array
def extractFileNameMetadata(fileList):
    
    filenameMetadata = numpy.chararray((numberOfFiles,6), itemsize=30)

    # expected format A is 'N:\\FannyGeorgi\\1-9_Virusprep\\1-9-4_Virusprep_HAdV-C5_dl312_Alexa488\\20160705-HAdV-dl312-pre-post-1dpi_Plate_2917\\Timepoint_1\\20160126-corning-all-spheroids-p1-003hps_A01_w1.TIF'
    # define the regex pattern A
    #filePattern = re.compile(r'.+\\Timepoint_(?P<timePoint>\d+)\\(?P<date>\d+)-(?P<experimentName>.*)-p(?P<plate>\d+)-(?P<hps>\d+)hps_(?P<wellID>\w\d\d)_w(?P<channel>\d+).TIF', re.IGNORECASE)

    # expected format B is 'C:\\Users\\FannyG\\Documents\\Data\\20151217_1-12_TumorRemission\\MorphoSphereTest\\1\\0dps\\20160126-corning-all-spheroids-p1-003hps_A01_w1.TIF'
    # define regex pattern B
    filePattern = re.compile(r'.+\\(?P<date>\d+)-(?P<experimentName>.*)-p(?P<plate>\d+)-(?P<hps>\d+)hps_(?P<wellID>\w\d\d)_w(?P<channel>\d+).TIF', re.IGNORECASE)

    j = 0

    for i in filenameData:
        match = filePattern.search(i)
        #print match.group('experimentName')
        ####### insert for loop to select only certain files into additional array selectedfiles
        ####### add array containing full paths
        filenameMetadata[j, 0] = match.group('date')
        #print filenameMetadata[j, 0]
        filenameMetadata[j, 1] = match.group('experimentName')
        filenameMetadata[j, 2] = match.group('plate')
        filenameMetadata[j, 3] = match.group('hps')
        filenameMetadata[j, 4] = match.group('wellID')
        filenameMetadata[j, 5] = match.group('channel')
        j = j+1

    #print filenameMetadata
    
    return filenameMetadata
