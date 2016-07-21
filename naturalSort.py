
# coding: utf-8

# In[16]:

#! python2
# naturalSort.py - natural sort of string list arrays

from natsort import natsorted


# In[17]:

def naturalSort(unsortedList=range(11)):
    if not isinstance(unsortedList, (list, tuple)):
        print('Error: The input list is invalid')
        return None
    sortedList=natsorted(unsortedList)
    
    return sortedList

