"""
datasetCreatorGUI: GUI to build dataset for MorphoSphere
@authors: Fanny Georgi, Vardan Andriasyan, Artur Yakimovich

"""

# UI for preparing training set for the CNN
# to be invoked in Jupyter
import time
from os import listdir, walk
from IPython.display import Image, display, clear_output
from ipywidgets import widgets


def submit_results(results, path):
    writer = csv.writer(open(path + 'results.csv', 'wb'))
    for key, value in results.items():
        writer.writerow([key, value])
    print("results saved to:" + path + 'results.csv')


def getImageFiles(path, pattern):
    files = []

    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    imageFiles = [f for f in files if re.match(pattern, f)]
    return imageFiles


def redraw(choices, x, i):
    display(Image(imagesPath + x[i]))
    time.sleep(1)

    buttons = [widgets.Button(description=choice) for choice in choices]

    container = widgets.HBox(children=buttons)
    display(container)

    def on_button_clicked(b):
        # [insert code to record choice]
        value[x[i]] = b.description
        container.close()
        clear_output()
        if (i <= len(x) - 2):
            redraw(choices, x, i + 1)
        else:
            print(value)
            submit_results(value, imagesPath)

    for button in buttons:
        button.on_click(on_button_clicked)


global value
global imagesPath

# define constants
imagesPath = 'C:\\Users\\Artur\\Documents\\images_test\\'
extension = "jpg"
choices = ['spheroid', 'non-spheroid']

# invoking
value = {}
imagesList = getImageFiles(imagesPath, re.compile(r".*(" + extension + ")"))
print('Number of Images to classify: ' + str(len(imagesList)))

redraw(choices, imagesList, 0)  # initializes the first choice