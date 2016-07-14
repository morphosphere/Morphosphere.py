
# coding: utf-8

# In[75]:

#!python2
# getFileList.py - returns absolute paths of all files in a folder
# including subfolders. naturally sorted according to the basenames.

# import packages
import os, natsort


# In[76]:

# get the input directory from a string
inputDirectory='C:\\Users\\LucaM\\envs\\py27\\share'


# In[89]:

# get basenames of all files in a directory (including subdirectories)
i=0
filenameList=[]
for foldernames, subfolders, filenames in os.walk(inputDirectory):
    for filename in filenames:
        #print(os.path.join(foldernames, filename))
        filenameList.append(os.path.join(foldernames, filename))
        print(filename)


# In[91]:

# sort the basenames with natsort
baseNames=[]
for i in filenameList:
    baseNames.append(os.path.basename(i))
    
sortedBasenames=natsort.natsorted(baseNames)

for i in sortedBasenames:
    print(i)


# In[92]:

# Ok, this is probably not the way one would do it..
# any suggestions to do do it properly?

filenameListSorted=[]
for i in sortedBasenames:
    for n in filenameList:
        if n.endswith(i):
            filenameListSorted.append(n)
            
for i in filenameListSorted:
    print(i)


# In[ ]:



