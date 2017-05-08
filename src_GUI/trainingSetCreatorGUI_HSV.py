# coding=utf-8
"""
TrainingSetCreatorGUI: GUI to build data set for MorphoSphere
@authors: Fanny Georgi, Artur Yakimovich,  Vardan Andriasyan

"""
########################################################################################################################
# Prerequisites for running in Anaconda environment:
# - SIP comes with Anaconda
# - Install PyQt: run 'conda install -c anaconda pyqt=4.11.4', then 'conda update --all' to get latest version
# - QT Designer comes with Anaconda
# - Build .ui file using Qt Designer: …\Anaconda2\Library\bin\designer-qt4.exe.

########################################################################################################################
# To do (indicated by ##):
# - Include back button
# - Function checking if all sets and classes are filled, then save
# - Make prettier buttons (e.g. images)
# - Add Tetris music
# - Count images by length of directory list instead of counter to ease assembly in multiple sessions.
# - Perform check if image directory exists
# - Print number of images per set and class after every assignment or after key input
# - Replace .ui by soft-coded button design
# - Build save option
# - Build load option
# - Correct positioning of sample images --> How do we generate sample images in a soft-coded manner?
# - Define user input in pop-up window

########################################################################################################################
import sys
from PyQt4 import QtCore, QtGui, uic
import cv2
from os import walk, path, makedirs
import re
from random import shuffle
import numpy as np
import datetime

########################################################################################################################
print str(datetime.datetime.now()) + ': Started MorphoSphere.'

# Define variables
## make this an optional pop-up window
imagesDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete\\4\\'
outputDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete_Analysis'
analysisTitle ='20160504_HSVc'
extension = "TIF"

# Load dataTable
dataTable = np.genfromtxt(outputDirectory + '\\' + analysisTitle + '.csv', dtype = object, delimiter =',', skip_header = 1)
dataTable = dataTable.astype(str)
dataTable = np.char.strip(dataTable)

# Paste images with flagFocus = 1 in new array
dataTableSelected = np.zeros((1,21), dtype= object)
for currentRow in range(0,len(dataTable),1):
    line = dataTable[currentRow, 11]
    if dataTable[currentRow, 11] == '1':
        newRow = dataTable[currentRow, :]
        dataTableSelected = np.vstack([dataTableSelected, newRow])
dataTableSelected = np.delete(dataTableSelected, (0), axis=0)

# Shuffle and save new array
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
np.take(dataTableSelected, np.random.rand(dataTableSelected.shape[0]).argsort(), axis=0, out= dataTableSelected)
selectedTitle = '_focused'
csvHeader = 'imagePath, uniqueID, plate, row, column, hps, dps, experimentNumber, date, channel, flagTL, flagFocus, flagWell, flagTraining, class, set, setClassCounter, imageHeight, imageWidth, segmentedImage, reducedImage'
outputDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete_Analysis'
np.savetxt(outputDirectory + '\\' + analysisTitle + selectedTitle +'.csv', dataTableSelected, fmt='%5s', delimiter=',', header=csvHeader )
classifiedTitle = '_classified'

# Directory checks
if not path.exists(imagesDirectory):
    print 'WARNING: The defined input directory does not exist. MorphoShere will terminate.'
    exit()
if not path.exists(outputDirectory):
    makedirs(outputDirectory)

# Define data sets
sets = ['Training','Test','Validation']

# Define class options
classes = ['healthy_spheroid', 'healthy_non-spheroid', 'unhealthy_spheroid', 'dead_spheroid', 'exclude']

# Define number of images per set in array
imagesTrainingSet = 10
imagesTestSet = 2 * imagesTrainingSet
imagesValidationSet = 2 * imagesTrainingSet

# Check if data set directories exist and require the user to confirm to proceed if it exists
# outputPathCheck = outputDirectory+'\\'+sets[0]+'\\'+classes[0]
# if path.exists(outputPathCheck):
#     print 'QUERY: The defined output directories already exist. To proceed anyway, press [y][Enter], else [n][Enter] to terminate.'
#     proceedFlag = raw_input().lower()
#     if proceedFlag == 'y':
#         print 'Proceeding.'
#     if proceedFlag == 'n':
#         print 'WARNING: MorphoSphere was terminated by the user.'
#         exit()
#     else:
#         print 'QUERY: The defined output directories already exist. To proceed anyway, press [y], else [n] to terminate.'
#
# if not path.exists(outputPathCheck):
#     # Create necessary directories
#     for iSet in sets:
#         for iClass in classes[:-1]:
#             outputPath = outputDirectory + '\\' + iSet + '\\' + iClass
#             if not path.exists(outputPath):
#                 makedirs(outputPath)
#
# # Create separate folder for excluded images, disable if desired
# excludePath = outputDirectory + '\\' + classes[-1]
# if not path.exists(excludePath):
#     makedirs(excludePath)
#
# print str(datetime.datetime.now()) + ': Created all necessary output directories.'

########################################################################################################################
# Generate list of images to classify from selected input directory
# print str(datetime.datetime.now()) + ': Starting to assemble list of images for manual classification from ' + imagesDirectory + '.'
#
# def getImageFiles(imagesPath, pattern):
#     fileList = []
#
#     for folders, subfolders, filenames in walk(imagesPath):
#         for filename in filenames:
#             if re.match(pattern, filename):
#                 fileList.append(path.join(folders, filename))
#
#     shuffle(fileList)
#     return fileList
#
# imagesList = getImageFiles(imagesDirectory, re.compile(r".*" + extension))
# print str(datetime.datetime.now()) + ': ' + str(len(imagesList)) + ' in ' + imagesDirectory + ' found for manual classification.'

print str(datetime.datetime.now()) + ': ' + str(len(dataTableSelected)) + ' focused TL images in ' + imagesDirectory + ' found for manual classification.'

########################################################################################################################
# Open GUI
qtCreatorFile = 'trainingSetGenerator.ui'

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.setWindowTitle('MorphoSphere')

        self.logo.setPixmap(QtGui.QPixmap('..\\img\\Logo_A_150.png'))

        headerColor = QtGui.QPalette()
        headerColor.setColor(QtGui.QPalette.Foreground, QtCore.Qt.white)
        self.header.setPalette(headerColor)

        # Show parameters in Welcome text
        ## Make font size dynamical
        self.welcomeText.setText('Welcome to the MorphoSphere Data Set Generator. \nThere are ' + str(len(dataTableSelected)) +' images from ' +
                                 imagesDirectory + ' ready to be manually classified as ' + classes[0] + ', ' + classes[1] + ', ' + classes[2] +
                                 ', ' + classes[3] +' or ' + classes[4] + '. You will be shown random images until ' +  str(imagesTrainingSet) +
                                 ' training and ' + str(imagesValidationSet) +
                                 ' validation and validation images have been selected for each data set and copied to ' + outputDirectory + '.')

        # Load data by making it self.
        self.dataTableSelected = dataTableSelected

        # Sample images
        ## Correct positioning
        self.class1Sample.setPixmap(QtGui.QPixmap('..\\img\\sample_SpheroidHealthy.png'))
        self.class2Sample.setPixmap(QtGui.QPixmap('..\\img\\sample_NonspheroidHealthy.png'))
        self.class3Sample.setPixmap(QtGui.QPixmap('..\\img\\sample_SpheroidUnhealthy.png'))
        self.class4Sample.setPixmap(QtGui.QPixmap('..\\img\\sample_NonspheroidDead.png'))

        # Image counters
        self.imageCounter = 0
        self.imageCounterClass1 = 0
        self.imageCounterClass2 = 0
        self.imageCounterClass3 = 0
        self.imageCounterClass4 = 0

        # Define button actions
        self.class1Button.clicked.connect(self.class1Button_clicked)
        self.class2Button.clicked.connect(self.class2Button_clicked)
        self.class3Button.clicked.connect(self.class3Button_clicked)
        self.class4Button.clicked.connect(self.class4Button_clicked)
        self.class5Button.clicked.connect(self.class5Button_clicked)

        # Define button text and shortcuts
        print 'HELP: Assign the image shown to a class by pressing the shortcut "Alt" + the "number" displayed on the button.'
        self.class1Button.setText('&1 ' + classes[0]) # 'Alt' + '1'
        self.class2Button.setText('&2 ' + classes[1]) # 'Alt' + '2'
        self.class3Button.setText('&3 ' + classes[2]) # 'Alt' + '3'
        self.class4Button.setText('&4 ' + classes[3]) # 'Alt' + '4'
        self.class5Button.setText('&5 ' + classes[4]) # 'Alt' + '5'

        # Slider
        self.labelFactor.setPalette(headerColor)
        self.labelAddend.setPalette(headerColor)

        # Initialize first image
        self.height = 420
        self.width = 420
        numberOfImagesNeeded = (len(classes)-1)  *  len(sets) * 5/3 * imagesTrainingSet

        if len(self.dataTableSelected) < numberOfImagesNeeded:
            print "WARNING: There are not enough images to fill all sets."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_listTooShort.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))

        if len(self.dataTableSelected) >= numberOfImagesNeeded:
            ## check if image exists
            self.displayImage()

        # Save status
        self.saveButton.setIcon(QtGui.QIcon('..\\img\\logo_save.png'))
        self.saveButton.clicked.connect(self.saveButton_clicked)
        ## save all parameters and lists
        ## include directory parser window

        # Quit
        self.cancelButton.setIcon(QtGui.QIcon( '..\\img\\logo_cancel.png'))
        self.cancelButton.clicked.connect(self.cancelButton_clicked)

        # Load
        self.loadButton.setIcon(QtGui.QIcon( '..\\img\\logo_load.png'))
        ## load saved parameters
        ## include directory parser window

    def factor_changed(self):
        newFactor = self.sliderFactor.value()
        self.scaleFactor = float(newFactor)
        self.imageFactor = cv2.multiply(self.imageResized, np.array([self.scaleFactor]))
        self.imageAddend = cv2.add(self.imageFactor, np.array([self.scaleAddend]))
        self.imageRGB = cv2.cvtColor(self.imageAddend, cv2.COLOR_GRAY2RGB)
        self.qImage = QtGui.QImage(self.height, self.width, QtGui.QImage.Format_RGB32)
        for x in xrange(self.width):
            for y in xrange(self.height):
                self.qImage.setPixel(x, y, QtGui.QColor(*self.imageRGB[x][y]).rgb())
        self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap(self.qImage).scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))

    def addend_changed(self):
        newAddend = self.sliderAddend.value()
        self.scaleAddend = float(newAddend)
        self.imageAddend = cv2.add(self.imageFactor, np.array([self.scaleAddend]))
        self.imageRGB = cv2.cvtColor(self.imageAddend, cv2.COLOR_GRAY2RGB)
        self.qImage = QtGui.QImage(self.height, self.width, QtGui.QImage.Format_RGB32)
        for x in xrange(self.width):
            for y in xrange(self.height):
                self.qImage.setPixel(x, y, QtGui.QColor(*self.imageRGB[x][y]).rgb())
        self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap(self.qImage).scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))

    def displayImage(self):
        image = cv2.imread(str(self.dataTableSelected[self.imageCounter,0]), 0)  # 0=grey, 1=RGB, -1=unchanged
        self.imageResized = cv2.resize(image, (self.height, self.width), interpolation=cv2.INTER_CUBIC)
        self.scaleFactor = 5.0
        self.sliderFactor.sliderReleased.connect(self.factor_changed)
        self.scaleAddend = 0.0
        self.sliderAddend.valueChanged.connect(self.addend_changed)
        self.imageFactor = cv2.multiply(self.imageResized, np.array([self.scaleFactor]))
        self.imageAddend = cv2.add(self.imageFactor, np.array([self.scaleAddend]))
        self.imageRGB = cv2.cvtColor(self.imageAddend, cv2.COLOR_GRAY2RGB)
        self.qImage = QtGui.QImage(self.height, self.width, QtGui.QImage.Format_RGB32)
        for x in xrange(self.width):
            for y in xrange(self.height):
                self.qImage.setPixel(x, y, QtGui.QColor(*self.imageRGB[x][y]).rgb())
        self.imageDisplay.setPixmap(
            QtGui.QPixmap(QtGui.QPixmap(self.qImage).scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))

        # Create unique image name
        # self.imagesDirectoryPattern = re.sub('\\\\', '_', imagesDirectory)
        # self.imagePathPattern = re.sub('\\\\', '_', dataTableSelected[self.imageCounter])
        # self.imageName = re.sub(self.imagesDirectoryPattern, '', self.imagePathPattern)

    def class1Button_clicked(self):
        print 'Image ' + self.dataTableSelected[self.imageCounter,1] + ' is image number ' + str(self.imageCounterClass1 + 1) + ' classified as ' + classes[0] + '.'
        self.dataTableSelected[self.imageCounter, 14] = classes[0]

        if self.imageCounterClass1 < imagesTrainingSet:
            self.dataTableSelected[self.imageCounter,15] = sets[0]

        if imagesTrainingSet <= self.imageCounterClass1 < (3*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[1]

        if (3*imagesTrainingSet) <= self.imageCounterClass1 < (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[2]

        if self.imageCounterClass1 >= (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 14] = 0
            print "Don't need more images for " + classes[0]

        # Test if end of image list reached
        if self.imageCounter == (len(self.dataTableSelected) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(self.dataTableSelected) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass1 = self.imageCounterClass1 + 1

            # Show next image from list
            self.displayImage()

    def class2Button_clicked(self):
        print 'Image ' + self.dataTableSelected[self.imageCounter,1] + ' is image number ' + str(self.imageCounterClass2 + 1) + ' classified as ' + classes[1] + '.'
        self.dataTableSelected[self.imageCounter, 14] = classes[1]

        if self.imageCounterClass2 < imagesTrainingSet:
            self.dataTableSelected[self.imageCounter, 15] = sets[0]

        if imagesTrainingSet <= self.imageCounterClass2 < (3*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[1]

        if (3*imagesTrainingSet) <= self.imageCounterClass2 < (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[2]

        if self.imageCounterClass2 >= (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 14] = 0
            print "INFO: Don't need more images for " + classes[1]

        # Test if end of image list reached
        if self.imageCounter == (len(self.dataTableSelected) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(self.dataTableSelected) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass2 = self.imageCounterClass2 + 1

            # Show next image from list
            self.displayImage()

    def class3Button_clicked(self):
        print 'Image ' + self.dataTableSelected[self.imageCounter, 1] + ' is image number ' + str(self.imageCounterClass3 + 1) + ' classified as ' + classes[2] + '.'
        self.dataTableSelected[self.imageCounter, 14] = classes[2]

        if self.imageCounterClass3 < imagesTrainingSet:
            self.dataTableSelected[self.imageCounter, 15] = sets[0]

        if imagesTrainingSet <= self.imageCounterClass3 < (3*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[1]

        if (3*imagesTrainingSet) <= self.imageCounterClass3 < (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[2]

        if self.imageCounterClass3 >= (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 14] = 0
            print "INFO: Don't need more images for " + classes[2]

        # Test if end of image list reached
        if self.imageCounter == (len(self.dataTableSelected) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(self.dataTableSelected) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass3 = self.imageCounterClass3 + 1

            # Show next image from list
            self.displayImage()

    def class4Button_clicked(self):
        print 'Image ' + self.dataTableSelected[self.imageCounter, 1] + ' is image number ' + str(self.imageCounterClass4 + 1) + ' classified as ' + classes[3] + '.'
        self.dataTableSelected[self.imageCounter, 14] = classes[3]

        if self.imageCounterClass4 < imagesTrainingSet:
            self.dataTableSelected[self.imageCounter, 15] = sets[0]

        if imagesTrainingSet <= self.imageCounterClass4 < (3*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[1]

        if (3*imagesTrainingSet) <= self.imageCounterClass4 < (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 15] = sets[2]

        if self.imageCounterClass4 >= (5*imagesTrainingSet):
            self.dataTableSelected[self.imageCounter, 14] = 0
            print "INFO: Don't need more images for " + classes[3]

        # Test if end of image list reached
        if self.imageCounter == (len(self.dataTableSelected) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(self.dataTableSelected) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass4 = self.imageCounterClass4 + 1

            # Show next image from list
            self.displayImage()

    def class5Button_clicked(self):
        print 'Image ' + self.dataTableSelected[self.imageCounter, 1] + ' was classified as ' + classes[4] + '.'
        self.dataTableSelected[self.imageCounter, 11] = 0
        self.dataTableSelected[self.imageCounter, 14] = classes[4]
        self.dataTableSelected[self.imageCounter, 15] = classes[4]

        # Test if end of image list reached
        if self.imageCounter == (len(self.dataTableSelected) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(self.dataTableSelected) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            # Numbers of images in class 5 is not tracked

            # Show next image from list
            self.displayImage()

    def cancelButton_clicked(self):
        print 'WARNING: MorphoSphere was terminated by the user.'
        self.close()

    def saveButton_clicked(self):
        print 'MorphoSphere was terminated by the user. All data will be saved to ' + outputDirectory + '.'

        # Strip table of spaces
        self.dataTableSelected = self.dataTableSelected.astype(str)
        self.dataTableSelected = np.char.strip(self.dataTableSelected)

        # Paste images with flagFocus = 1 in new array
        dataTableClassified = np.zeros((1, 21), dtype=object)
        for currentRow in range(0, len(self.dataTableSelected), 1):
            line = self.dataTableSelected[currentRow, 11]
            if self.dataTableSelected[currentRow, 11] == '1':
                newRow = self.dataTableSelected[currentRow, :]
                dataTableClassified = np.vstack([dataTableClassified, newRow])
        dataTableClassified = np.delete(dataTableClassified, (0), axis=0)

        np.savetxt(outputDirectory + '\\' + analysisTitle + classifiedTitle + '.csv', dataTableClassified, fmt='%5s',delimiter=',', header=csvHeader)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    # Fix window size to impair resizing
    window.setFixedSize(980, 900)
    # Set background
    background = QtGui.QPalette()
    background.setColor(QtGui.QPalette.Background, QtCore.Qt.darkGray)
    window.setPalette(background)
    sys.exit(app.exec_())

########################################################################################################################
print "this is for the breakpoint"