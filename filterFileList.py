
# coding: utf-8

# In[9]:

#! python2
# filterFileList.py - filter the filenameList according to a regex

import re


# In[23]:

def filterFileList(fileList, regex=r'_(?P<siteName>s\d*)_(?P<channelName>w\d*).TIF'):
    
    # validate the regex
    try:
        regex=re.compile(regex)
        is_valid=True
        print('Filtering...')
    except re.error:
        print('The entered regex is invalid.')
        is_valid=False
        return None
    
    # filter the input list
    filteredList=[]
    for fileName in fileList:
        if regex.search(fileName):
            filteredList.append(fileName)
    
    print('Done.')
    return filteredList

