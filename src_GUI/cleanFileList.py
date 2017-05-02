# coding=utf-8
"""
cleanFileList: Assemble list of focused transmission light images for MorphoSphere
based on extractMetadata.py - extracts metadata from TIF file descriptions
use parseImageDescription(fileName)
returns a dictionary with the file attributes (e.g. {'nExposure': '35 ms' ...})
@authors: Fanny Georgi, Luca Murer
"""
########################################################################################################################
# Prerequisites for running in Anaconda environment:


########################################################################################################################
# To do (indicated by ##):
# - Exclude unfocused images based on focusing attempts
# - Select TL images based on metadata
# - Write output cvs
# - Make nice functions

########################################################################################################################
# Imports
from os import walk, path
from random import shuffle
import re
import numpy
import natsort
import cv2

########################################################################################################################
# Read images from directory
imagesDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete\\'
analysisTitle ='20160501_Test2'

# Exclude thumbnails
# Select clean [number]dps folders, TIFs only
pattern = r"^.*w\d((?!_Thumb).)*.TIF$"
#pattern = r"^.*\\\\\d+\\\\\d+dps\\.((?!Thumb).)*.TIF$"


def getImageFiles(imagesPath, pattern):
    fileList = []

    for folders, subfolders, filenames in walk(imagesPath):
        for filename in filenames:
            if re.match(pattern, path.join(folders, filename)):
                fileList.append(path.join(folders, filename))

    shuffle(fileList)
    return fileList

imagesList = getImageFiles(imagesDirectory, re.compile(pattern))
imagesList = natsort.natsorted(imagesList)

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
csvHeader = 'imagePath, uniqueID, plate, row, column, hps, dps, experimentNumber, date, channel, flagTL, flagFocus, flagWell, flagTraining, class, set, setClassCounter, imageHeight, imageWidth, segmentedImage, reducedImage'

dataTable = numpy.zeros((len(imagesList),21), dtype= object)
dataTable[:,0] = imagesList

# Iterate over dataTable[:,0] to extract filename-related info
## Later extract from metadata
for image in range(0,len(dataTable),1):

    # Extract file name information, ***NEEDS TO BE CUSTOMIZED***
    regexFileName = re.search('^.*\\\\\d{8}-(?P<experimentNumber>\d+-\d+-\d+)_.*(?P<timePointDps>[0-9]+)dps\\\\(?P<date>\d{8}).*-p(?P<plate>\d+)-(?P<timePointHps>[0-9][0-9][0-9])hps_(?P<row>[A-Z])(?P<column>[0-9][0-9])_w(?P<channel>[0-9]).TIF$', dataTable[image,0])
    #regexFileName = re.search('^.*\\\\.*(\d+-\d+-\d+)_.*\\\\(\d+)dps\\\\(\d{8}).*-p(\d+)-(\d+)hps_([A-Z])(\d\d)_w(\d).*.TIF$', dataTable[image,0])
    print dataTable[image,0]
    dataTable[image,2] = regexFileName.group('plate')
    dataTable[image,3] = regexFileName.group('row')
    dataTable[image,4] = regexFileName.group('column')
    dataTable[image,5] = regexFileName.group('timePointHps')
    dataTable[image,6] = regexFileName.group('timePointDps')
    dataTable[image,7] = regexFileName.group('experimentNumber') +'-' # prevent reading as data
    dataTable[image,8] = regexFileName.group('date')
    dataTable[image,9] = regexFileName.group('channel')

    # Create unique identifier
    dataTable[image, 1] = 'p' + dataTable[image,2] + '_t' + dataTable[image,5] + '_w' + dataTable[image,3] + dataTable[image,4] + '_c' + dataTable[image,9]

    # Select only certain wells, time points etc. ***NEEDS TO BE CUSTOMIZED***
    #pattern = r'^.*(?P<timePointDps>[0-9]+)dps\\\\(?P<date>\d{8}).*p(?P<plate>\d+)-(?P<timePointHps>[0-9][0-9][0-9])hps_(?P<row>[A-P])(?P<column>[0][1-6])_w(?P<channel>[1])((?P<exclude>!_Thumb).)*.TIF$'
    rowPattern = re.compile('[A-P]')
    columnPattern = re.compile('[0][1-6]')
    dpsPattern = re.compile('\d+')
    if rowPattern.match(dataTable[image,3]) and columnPattern.match(dataTable[image,4]) and dpsPattern.match(dataTable[image,6]):
        dataTable[image, 12] = 1

########################################################################################################################
# Select only transmission light images

# Define variables
#imagesDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete\\TestCleanFileList\\'
#imageForTest = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete\\TestCleanFileList\\2\\4dps\\20160130-corning-all-spheroids-p2-095hps_A01_w1.TIF'

### Dirty image-based approach
for currentImage in range(0,len(dataTable),1):
    if dataTable[currentImage,12] == 1:
        imageToAnalyze = cv2.imread(dataTable[currentImage, 0], -1)  # 0 = grey, 1=RGB, -1=unchanged
        # Analyze mean intensity of image corner to avoid influence of transgene-associated expression
        imageCropped = imageToAnalyze[0:100, 0:100]
        if numpy.max(cv2.mean(imageCropped)) > 4000:
            dataTable[currentImage,10] = 1
        else:
            dataTable[currentImage, 10] = 0

### Luca's approach, DOES NOT READ ALL INFO
# def parseImageDescription(fileName):
#     # open the image
#     image = Image.open(fileName)
#
#     # extract image description (title)
#     tags = str(image.tag[270])
#
#     # define the regex to parse the description
#     regex = re.compile(r'(?P<key>[-\w\s\d.,()\/_]*):(?P<value>[-\w\s\d.,()\/_]*)\\r\\n')
#
#     # generate a dictionary with the matches
#     tagDict = {}
#     for match in regex.finditer(tags):
#         # remove whitespace at the beginning of the values
#         value = match.group('value')
#         if value.startswith(' '):
#             value = value[1:]
#         # assign keys and values
#         tagDict[match.group('key')] = value
#
#     return tagDict
#
#testDict = parseImageDescription(imageForTest)

### tifffile
# import tifffile
# with tifffile.TiffFile(imageForTest) as tif:
#     images = tif.asarray()
#     for page in tif:
#         for tag in page.tags.values():
#             t = tag.name, tag.value
#         image = page.asarray()

### Bioformats
# following http://pythonhosted.org/python-bioformats/http://pythonhosted.org/python-bioformats/
# - install new version Java Development Kit (JDK), install Microsoft Visuals 2017, set PATHs to python there
# - install javabridge binary from http://www.lfd.uci.edu/%7Egohlke/pythonlibs/#javabridge in VS 2017
# - all xmls etc saved in sample data folder
# channel in Image.Pixels.Channel.Name = "TL20", access via
# o = OMEXML()
# o.image().Pixels. ...

#### Working example
# import javabridge
# import bioformats
# path = r"C:\\Users\\FannyGeorgi\\Desktop\\multi-channel-time-series.ome.tif"
#
# javabridge.start_vm(class_path=bioformats.JARS)
# rdr = javabridge.JClassWrapper('loci.formats.in.OMETiffReader')()
# rdr.setOriginalMetadataPopulated(True)
# clsOMEXMLService = javabridge.JClassWrapper('loci.formats.services.OMEXMLService')
# serviceFactory = javabridge.JClassWrapper('loci.common.services.ServiceFactory')()
# service = serviceFactory.getInstance(clsOMEXMLService.klass)
# metadata = service.createOMEXMLMetadata()
# rdr.setMetadataStore(metadata)
# rdr.setId(path)
# root = metadata.getRoot()
# first_image = root.getImage(0)
# pixels = first_image.getPixels()
# # The plane data isn't in the planes, it's in the tiff data
# for idx in range(pixels.sizeOfTiffDataList()):
#     tiffData = pixels.getTiffData(idx)
#     c = tiffData.getFirstC().getValue().intValue()
#     t = tiffData.getFirstT().getValue().intValue()
#     print "TiffData: c=%d, t=%d" % (c, t)
#
#### Actual code
# import javabridge
# import bioformats
# import xml.etree.ElementTree as ET
#
# path = r"N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete\\TestCleanFileList\\2\\4dps\\20160130-corning-all-spheroids-p2-095hps_A01_w1.TIF"
# javabridge.start_vm(class_path=bioformats.JARS)
# # default schema in omexml.py is 2013-06, reads with 2015-03
# omeXml = bioformats.get_omexml_metadata(path)
# omeXmlString = omeXml.encode('utf-8')
#
# tree = ET.fromstring(omeXmlString)
# # Define namespace
# nameSpace = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
# originalMetadata = tree.findall(".//{}OriginalMetadata".format(nameSpace))
# # Iterate in founded origins
# for origin in originalMetadata:
#     key = origin.find("{}Key".format(nameSpace)).text
#     if key == "Information|Image|S|Scene|Shape|Name": ## needs correction here
#          value = origin.find("{}Value".format(nameSpace)).text
#          print("Value: {}".format(value))
#
# javabridge.kill_vm()

### PIL
# from PIL import Image
# im = Image.open(imageForTest)
# imageDesc = im.tag[270]
# print imageDesc
# for t in im.tag.keys():
#     print t, im.tag[t]

########################################################################################################################
# Select best-focused image

for currentImage in range(0,len(dataTable),1):
    # Only analyze selected wells and TL images
    if dataTable[currentImage,12] == 1 and dataTable[currentImage,10] == 1:

        # Check only if well has not been analyzed (flag will be changed to yes/no)
        if dataTable[currentImage,11] == 0:
            dataTable[currentImage,11] = 2
            # Create new array for convenient comparison of sharpness containing path, line in dataTable and sharpness
            wellList = numpy.zeros((1,3), dtype=object)
            wellList[0,0] = dataTable[currentImage,0]
            wellList[0,2] = currentImage

            imageToAnalyze = cv2.imread(wellList[0, 0], -1)  # 0 = grey, 1=RGB, -1=unchanged
            sobelx = cv2.Sobel(imageToAnalyze, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(imageToAnalyze, cv2.CV_64F, 0, 1, ksize=5)
            sobelSqrt = numpy.sqrt(numpy.multiply(sobelx, sobelx) + numpy.multiply(sobely, sobely))
            wellList[0, 1] = sum(numpy.divide(sum(sobelSqrt), sobelx.size))

            # Find max number of channels to check this number -1 of images below
            maxNumberOfChannels = int(numpy.amax(dataTable[:,9]))
            for imagesBelow in range(1,maxNumberOfChannels - int(dataTable[currentImage,9]) +1,1):
                # Find all images of same well and write paths into wellList
                if dataTable[currentImage + imagesBelow,10] == 1 and dataTable[(currentImage + imagesBelow), 2] == dataTable[currentImage, 2] and dataTable[(currentImage + imagesBelow), 3] == dataTable[currentImage, 3] and dataTable[(currentImage + imagesBelow), 4] == dataTable[currentImage, 4] and dataTable[(currentImage + imagesBelow), 5] == dataTable[currentImage, 5]:
                    newRow = numpy.zeros((1, 3), dtype=object)
                    newRow[0,0] = dataTable[(currentImage + imagesBelow),0]
                    # Keep track of row in dataTable index
                    newRow[0,2] = currentImage + imagesBelow

                    imageToAnalyze = cv2.imread(dataTable[(currentImage + imagesBelow),0], -1) #0 = grey, 1=RGB, -1=unchanged
                    sobelx = cv2.Sobel(imageToAnalyze, cv2.CV_64F, 1, 0, ksize=5)
                    sobely = cv2.Sobel(imageToAnalyze, cv2.CV_64F, 0, 1, ksize=5)
                    sobelSqrt = numpy.sqrt(numpy.multiply(sobelx, sobelx) + numpy.multiply(sobely, sobely))
                    newRow[0,1] = sum(numpy.divide(sum(sobelSqrt), sobelx.size))

                    wellList = numpy.concatenate((wellList,newRow))
                    dataTable[(currentImage + imagesBelow),11] = 2

            for wellListImage in range(0,len(wellList), 1):
                if wellList[wellListImage,1] == numpy.amax(wellList[:,1],axis=0):
                    dataTable[wellList[wellListImage,2],11] = 1
                else:
                    dataTable[wellList[wellListImage,2],11] = 2

########################################################################################################################
# Export table and include header
outputDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete_Analysis'
numpy.savetxt(outputDirectory + '\\' + analysisTitle+'.csv', dataTable, fmt='%5s', delimiter=',', header=csvHeader )

# a = numpy.array([[0.0,1.630000e+01,1.990000e+01,1.840000e+01],
#                  [1.0,1.630000e+01,1.990000e+01,1.840000e+01],
#                  [2.0,1.630000e+01,1.990000e+01,1.840000e+01]])
# fmt = ",".join(["%s"] + ["%10.6e"] * (a.shape[1]-1))
# numpy.savetxt("temp", a, fmt=fmt, header="SP,1,2,3", comments='')

########################################################################################################################
print "this is for the breakpoint"