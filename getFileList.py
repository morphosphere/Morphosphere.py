
# coding: utf-8

# In[2]:

#!python2
# getFileList.py - returns absolute paths of all files in a folder
# including subfolders. naturally sorted according to the basenames.

# import packages
import os, natsort


# In[53]:

# get basenames of all files in a directory (including subdirectories)


def getFileList(targetPath='.\\'):
    print('Getting files from: %s' % targetPath)
    
    if not os.path.isdir(targetPath):
        print('Path not found.')
        return None
    i=0
    filenameList=[]
    for foldernames, subfolders, filenames in os.walk(targetPath):
        for filename in filenames:
            #print(os.path.join(foldernames, filename))
            filenameList.append(os.path.join(foldernames, filename))
    filenameList=natsort.natsorted(filenameList)

    return filenameList

