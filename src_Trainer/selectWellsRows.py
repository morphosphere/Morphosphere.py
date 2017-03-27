
# coding: utf-8

#! python2
# selectWellsRows.py - takes a list of files as an input and returns the selected filenames
# The function matches the filenames only if the rows and columns are in the selection list.
# This means, that we can so far only select rectangular areas of adjacent wells, but not single wells

import re

def selectWells(fileList=None, rows=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                columns=[i+1 for i in range(12)],
                regex=r'(?P<wellName>(?P<rowName>\w)(?P<columnName>\d\d))_(?P<siteName>s\d*)_(?P<channelName>w\d*).TIF'):
                
    reObj=re.compile(regex)
    
    selectedFiles=[]
    
# validate fileList    
    try:
        len(fileList)>0
    except TypeError:
        print('input fileList invalid')
        return None

# filtering fileList
    for fileName in fileList:
        matches=reObj.search(fileName)
        for row in rows:
            if matches.group('rowName')==row:
                for column in columns:
                    if int(matches.group('columnName'))==int(column):
                        selectedFiles.append(fileName)

    return(selectedFiles)

