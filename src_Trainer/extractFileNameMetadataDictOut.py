# coding: utf-8

#! python2
# this script defines a function extractFileNameMetadata() that generates a
# dictionary with the metadata extracted from a filename according
# to a regular expression. Note that this works on single filename level.

# import packages
import re

# expected format is .\Timepoint_1\20160126-corning-all-spheroids-p1-003hps_A01_w1.TIF
def extractFileNameMetadata(fileName):
    
    # define the regex pattern
    fileNamePattern = re.compile(r'(?:Timepoint_(?P<timePoint>\d+))?\\(?P<date>\d{6,})-(?P<experimentName>.*)-p(?P<plate>\d+)(?:-(?P<hps>\d+)hps)?_(?P<wellID>\w\d\d)_w(?P<channel>\d+).TIF')
    
    # produce a match object
    match=fileNamePattern.search(fileName)
    
    # produce a dict of the match object
    fileNameMetadata={}
    fileNameMetadata=match.groupdict()
    
    return fileNameMetadata
