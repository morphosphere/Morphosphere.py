
# coding: utf-8

# In[123]:

#! python2
# extractMetadata.py - extracts metadata from TIF file descriptions
# use parseImageDescription(fileName)
# returns a dictionary with the file attributes (e.g. {'nExposure': '35 ms' ...})
from PIL import Image
import re


# In[126]:

def parseImageDescription(fileName):

    # open the image
    image=Image.open(fileName)

    # extract image description (title)
    tags=str(image.tag[270])

    # define the regex to parse the description
    regex=re.compile(r'(?P<key>[-\w\s\d.,()\/_]*):(?P<value>[-\w\s\d.,()\/_]*)\\r\\n')

    # generate a dictionary with the matches
    tagDict={}
    for match in regex.finditer(tags):
        # remove whitespace at the beginning of the values 
        value=match.group('value')
        if value.startswith(' '):
            value=value[1:]
        #assign keys and values
        tagDict[match.group('key')]=value
    
    return tagDict

