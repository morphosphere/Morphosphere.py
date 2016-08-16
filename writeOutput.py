# coding: utf-8
#!python2

# Written by ... for MorphoSphere, 2016

# This script defines how the results are assembled and saved into a csv file.
#??? Shall we save everything in px or um? Then we need to retrieve the conversion factor somewhere.
# Additionally, the segmented images can be saved as tiffs.
# Additionally, output plots can be predefined here.
#??? Shall we also save the trained CNN and the training results here?

# Input: inputDirectory as string
          # filenameMetadata as numberOfSelectedFiles x keys numpy character array
          # selectedFilenameList as one column numpy string array
          # tagDics as dictionary
          # segmentedClumps as numberOfSelectedFiles x keys numpy character array
          # classyfiedClumps as numberOfSelectedFiles x keys numpy character array
          # analyzedClumps as numberOfSelectedFiles x keys numpy character array

# Returns: 


# import packages
import os


def writeOutput(inputDirectory, selectedFilenameList, tagDict, filenameMetadata, classyfiedClumps):
    
    # We should have a outputName variable for this and then add suffixes for _csv, _plot and whatever
    outputName = ''
    
    # create an 'MorphoSphere_outputName' folder in inputDirectory\MorphoSphere_outputName
    # This makes multiple runs of MorphoSphere on the same experiment easier
    
    # Concatenate filenameMetadata = path\filename, date, experimentName, plate, hps, wellID, channel
    # with tagDict[number of focusing attempts = infocusFlag
    # with segmentedClumps = ROI coordinates
    # with classyfiedClumps = class, confidence (or whatever other relevant classification quality measures)
    # with analyzedClumps = arithmeticMean, MinorAxisLength, MajorAxisLength, area, eccentricity
    
    # save csv into inputDirectory\MorphoSphere_outputName using outputName_csv
    
    # create inputDirectory\MorphoSphere_outputName\Images
    # save tiffs of segemented spheroids using outputName_wellID or something like that
    # make sure they all have the same dimensions (max of all)
    # possibly save csv indicating change of pixel size
    
    # create inputDirectory\MorphoSphere_outputName\Plots
    # make fancy plots
    # save fancy plots using outputName_plotxyz
