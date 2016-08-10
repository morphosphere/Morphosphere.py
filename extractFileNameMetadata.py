
# coding: utf-8

# Written by Fanny Georgi and Luca Murer for MorphoSphere, 2016

# This script reads files from a folder architecture and generates an
# array with the metadata extracted from the filenames according to regular expressions. 
# Note that these need to be custom-adjusted for specific file architectures and filenames.

# If this script is converted into a function, the following variables need to be handed over: inputDirectory, filePattern
# If this script is converted into a function, the following variables need to be returned: filenameMetadata

# import packages
import os, re, numpy

# retrieve the input directory from a string
inputDirectory = 'C:\\Users\\FannyG\\Documents\\Data\\20151217_1-12_TumorRemission\\MorphoSphereTest'

# read basenames of all .TIF files in the input directory (including subdirectories) into array
filenameList = []
i = 0
tifFiles = re.compile(r'^.*[A-Z][0-9]*_w[0-9]*\.tif$', re.IGNORECASE)
for foldernames, subfolders, filenames in os.walk(inputDirectory):
    for filename in filenames:
        if tifFiles.match(filename) != None:
            filenameList.append(os.path.join(foldernames, filename))
            
###### insert sort function
            
# convert file list into 1 colon numpy string array, this makes regexing easier
filenameData = numpy.array(filenameList, dtype=str).T
#print filenameData.shape
#print filenameData
#print type(filenameData[0]
            
# display number of files detected
numberOfFiles = len(filenameList)
numberOfPlates = numberOfFiles/384
print "%s files have been detected, equal to %s plates." % (numberOfFiles, numberOfPlates)

# extract filename metadata into numpy character array

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
