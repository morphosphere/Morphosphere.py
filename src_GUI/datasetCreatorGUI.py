"""
datasetCreatorGUI: GUI to build dataset for MorphoSphere
@authors: Fanny Georgi, Vardan Andriasyan, Artur Yakimovich

"""

# Prerequisites:
# (-) Delete Thumbs from all folders to be included in the data pool
# - Ensure only raw data is in the specified input directory

# Running instructions:
# to be invoked in Jupyter (Fanny: not as Admin)


import time
from os import listdir, walk, path, makedirs
import glob
import re
from random import shuffle
from IPython.display import Image, display, clear_output
from ipywidgets import widgets


def submit_results(results, path):
    writer = csv.writer(open(path + 'results.csv', 'wb'))
    for key, value in results.items():
        writer.writerow([key, value])
    print("results saved to:" + path + 'results.csv')


def getImageFiles(imagesPath, pattern):
    fileList = []

    for folders, subfolders, filenames in walk(imagesPath):
        for filename in filenames:
            if re.match(pattern, filename):
                fileList.append(path.join(folders, filename))

    shuffle(fileList)
    return fileList

def redraw(choices, imagesList, i):
    display(Image(imagesList[i]))
    time.sleep(1)

    buttons = [widgets.Button(description=choice) for choice in choices]

    container = widgets.HBox(children=buttons)
    display(container)

    def on_button_clicked(b):
        # [insert code to record choice]
        value[imagesList[i]] = b.description
        container.close()
        clear_output()
        if (i <= len(imagesList) - 2):
            redraw(choices, imagesList, i + 1)
        else:
            print(value)
            submit_results(value, imagesPath)

    for button in buttons:
        button.on_click(on_button_clicked)


global value

# define constants
imagesPath = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20161111_1-12-6_DyingPhenotype\\'
outputPath = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\InputDataGUI\\'
extension = "TIF"
choices = ['healthy_spheroid', 'healthy_non-spheroid', 'unhealthy_spheroid', 'dead_spheroid', 'exclude: oof/ dirt/ non-TL']
sizeOfSet = 50 # number of images per set to be selected

# create output directories
choiceNumber = len(choices) - 1
choice = 0
while choice < choiceNumber:
    if not path.exists(outputPath + choices[choice]):
        makedirs(outputPath + choices[choice])
    choice = choice + 1

# invoking
value = {}
imagesList = getImageFiles(imagesPath, re.compile(r".*" + extension))
print('Number of Images to classify: ' + str(len(imagesList)))
#print "There are %d images ready for manual classification." % len(fileList)

redraw(choices, imagesList, 0)  # initializes the first choice