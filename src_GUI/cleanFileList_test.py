# coding=utf-8

import re

# Populate big data table
# col 0 = image path "imagePath"
# col 1 = unique identifier "uniqueID"
# col 2 = "plate"
# col 3 = "row"
# col 4 = "column"
# col 5 = timepoint "hps"
# col 6 = timepoint "dps"
# col 7 = "experimentNumber"
# col 8 = "date"
# col 9 = "channel"
# col 10 = "flagTL"
# col 11 = "flagFocus"
# col 12 = "flagWell"
# col 13 = selected for training flag
# col 14 = class
# col 15 = set
# col 16 = setClass counter
# col 17 = image height
# col 18 = image width
# col 19 = cropped segmented image
# col 20 = down sampled image
# Sum =  21 columns


image = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete\\1\\0dps\\20160126-corning-all-spheroids-p1-003hps_A01_w2.TIF'

regexFileName = re.search('^.*\\\\\d{8}-(?P<experimentNumber>\d+-\d+-\d+)_.*(?P<timePointDps>[0-9]+)dps\\\\(?P<date>\d{8}).*-p(?P<plate>\d+)-(?P<timePointHps>[0-9][0-9][0-9])hps_(?P<row>[A-Z])(?P<column>[0-9][0-9])_w(?P<channel>[0-9]).TIF$', image)
#regexFileName = re.search('^.*\\\\.*(\d+-\d+-\d+)_.*\\\\(\d+)dps\\\\(\d{8}).*-p(\d+)-(\d+)hps_([A-Z])(\d\d)_w(\d).*.TIF$', dataTable[image,0])
print regexFileName.group('plate')
plate = regexFileName.group('plate')
row = regexFileName.group('row')
column = regexFileName.group('column')
hps = regexFileName.group('timePointHps')
dps = regexFileName.group('timePointDps')
exNumber = regexFileName.group('experimentNumber') +'-' # prevent reading as data
date = regexFileName.group('date')
channel = regexFileName.group('channel')

print "this is for the breakpoint"