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
# - Count images by length of directory list instead of counter to ease assembly in multiple sessions.
# - Check if user input directories exist
# - Check if output directories are empty, if not, require user to proceed by pressing 'y'
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
print str(datetime.datetime.now()) + ': Started the MorphoSphere Data Set Generator.'

# Define variables
## make this an optional pop-up window
imagesDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20161111_1-12-6_DyingPhenotype\\'
outputDirectory = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\InputDataGUI_1-12-6'
extension = "TIF"
imagesPerSet = 100

# For brief testing only
#currentImagePath = ['..\\img\\20161111-1-12-6-003hps_C03_w1.TIF', '..\\img\\20161119-1-12-6-194hps_C03_w1.TIF', '..\\img\\20161205-1-12-6-654hps_M19_w1.TIF', '..\\img\\20161205-1-12-6-654hps_M17_w1.TIF', '..\\img\\20161118-1-12-6-168hps_J21_w1.TIF', '..\\img\\20161111-1-12-6-003hps_C03_w1.TIF', '..\\img\\20161119-1-12-6-194hps_C03_w1.TIF', '..\\img\\20161205-1-12-6-654hps_M19_w1.TIF', '..\\img\\20161205-1-12-6-654hps_M17_w1.TIF', '..\\img\\20161118-1-12-6-168hps_J21_w1.TIF', '..\\img\\20161111-1-12-6-003hps_C03_w1.TIF', '..\\img\\20161119-1-12-6-194hps_C03_w1.TIF', '..\\img\\20161205-1-12-6-654hps_M19_w1.TIF', '..\\img\\20161205-1-12-6-654hps_M17_w1.TIF', '..\\img\\20161118-1-12-6-168hps_J21_w1.TIF']
#currentImageName = ['test1.TIF', 'test2.TIF', 'test3.TIF', 'test4.TIF','test5.TIF', 'test6.TIF', 'test7.TIF', 'test8.TIF', 'test9.TIF','test10.TIF', 'test11.TIF','test12.TIF']

# Define class options
classes = ['healthy_spheroid', 'healthy_non-spheroid', 'unhealthy_spheroid', 'dead_spheroid', 'exclude']

# Directory checks
## check if imagesDirectory exists and is not empty, else give warning
## check if outputDirectory exists and is empty, else give warning and delete content

# Create necessary directories
sets = ['Test','Training','Validation']
for iSet in sets:
    for iClass in classes[:-1]:
        outputPath = outputDirectory+'\\'+iSet+'\\'+iClass
        if not path.exists(outputPath):
            makedirs(outputPath)
# Create separate folder for excluded images, disable if desired
excludePath = outputDirectory+'\\'+classes[-1]
if not path.exists(excludePath):
    makedirs(excludePath)

print str(datetime.datetime.now()) + ': Created all necessary output directories.'

########################################################################################################################
# Generate list of images to classify from selected input directory
print str(datetime.datetime.now()) + ': Starting to assemble list of images for manual classification from ' + imagesDirectory + '.'
global value
value = {}

def getImageFiles(imagesPath, pattern):
    fileList = []

    for folders, subfolders, filenames in walk(imagesPath):
        for filename in filenames:
            if re.match(pattern, filename):
                fileList.append(path.join(folders, filename))

    shuffle(fileList)
    return fileList

imagesList = getImageFiles(imagesDirectory, re.compile(r".*" + extension))
print str(datetime.datetime.now()) + ': ' + str(len(imagesList)) + ' in ' + imagesDirectory + ' found for manual classification.'

########################################################################################################################
# Open GUI
qtCreatorFile = 'trainingSetGenerator.ui'

Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.setWindowTitle('MorphoSphere Data Set Generator')

        self.logo.setPixmap(QtGui.QPixmap('..\\img\\Logo_A_150.png'))

        headerColor = QtGui.QPalette()
        headerColor.setColor(QtGui.QPalette.Foreground, QtCore.Qt.white)
        self.header.setPalette(headerColor)

        # Show parameters in Welcome text
        self.welcomeText.setText('Welcome to the MorphoSphere Data Set Generator. \nThere are ' + str(len(imagesList)) +' images from ' +
                                 imagesDirectory + ' ready to be manually classified as ' + classes[0] + ', ' + classes[1] + ', ' + classes[2] +
                                 ', ' + classes[3] +' or ' + classes[4] + '. You will be shown random images until ' +  str(imagesPerSet) +
                                 ' images have been selected for each data set and copied to ' + outputDirectory + '.')

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
        numberOfImagesNeeded = (len(classes)-1)  *  len(sets)

        if len(imagesList) < numberOfImagesNeeded:
            print "WARNING: There are not enough images to fill all sets."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_listTooShort.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))

        if len(imagesList) >= numberOfImagesNeeded:
            ## check if image exists
            self.displayImage()

        # Save status
        self.saveButton.setIcon(QtGui.QIcon('..\\img\\logo_save.png'))
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
        image = cv2.imread(imagesList[self.imageCounter], 0)  # 0=grey, 1=RGB, -1=unchanged
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
        self.imagesDirectoryPattern = re.sub('\\\\', '_', imagesDirectory)
        self.imagePathPattern = re.sub('\\\\', '_', imagesList[self.imageCounter])
        self.imageName = re.sub(self.imagesDirectoryPattern, '', self.imagePathPattern)

    def class1Button_clicked(self):
        print 'Image ' + imagesList[self.imageCounter] + ' is image number ' + str(self.imageCounterClass1 + 1) + ' classified as ' + classes[0] + '.'
        self.currentImage = cv2.imread(imagesList[self.imageCounter], 0)  # 0=grey, 1=RGB, -1=unchanged

        if self.imageCounterClass1 < imagesPerSet:
            cv2.imwrite(outputDirectory+'\\'+sets[0]+'\\'+classes[0]+'\\'+ self.imageName, self.currentImage)

        if imagesPerSet <= self.imageCounterClass1 < (2*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[1]+'\\'+classes[0]+'\\'+ self.imageName, self.currentImage)

        if (2*imagesPerSet) <= self.imageCounterClass1 < (3*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[2]+'\\'+classes[0]+'\\'+ self.imageName, self.currentImage)

        if self.imageCounterClass1 > (3*imagesPerSet):
            print "Don't need more images for " + classes[0]

        # Test if end of image list reached
        if self.imageCounter == (len(imagesList) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(imagesList) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass1 = self.imageCounterClass1 + 1

            # Show next image from list
            self.displayImage()

    def class2Button_clicked(self):
        print 'Image ' + imagesList[self.imageCounter] + ' is image number ' + str(self.imageCounterClass2 + 1) + ' classified as ' + classes[1] + '.'
        self.currentImage = cv2.imread(imagesList[self.imageCounter], 0)  # 0=grey, 1=RGB, -1=unchanged

        if self.imageCounterClass1 < imagesPerSet:
            cv2.imwrite(outputDirectory+'\\'+sets[0]+'\\'+classes[1]+'\\'+ self.imageName, self.currentImage)

        if imagesPerSet <= self.imageCounterClass1 < (2*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[1]+'\\'+classes[1]+'\\'+ self.imageName, self.currentImage)

        if (2*imagesPerSet) <= self.imageCounterClass1 < (3*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[2]+'\\'+classes[1]+'\\'+ self.imageName, self.currentImage)

        if self.imageCounterClass1 > (3*imagesPerSet):
            print "INFO: Don't need more images for " + classes[1]

        # Test if end of image list reached
        if self.imageCounter == (len(imagesList) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(imagesList) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass2 = self.imageCounterClass2 + 1

            # Show next image from list
            self.displayImage()

    def class3Button_clicked(self):
        print 'Image ' + imagesList[self.imageCounter] + ' is image number ' + str(self.imageCounterClass3 + 1) + ' classified as ' + classes[2] + '.'
        self.currentImage = cv2.imread(imagesList[self.imageCounter], 0)  # 0=grey, 1=RGB, -1=unchanged

        if self.imageCounterClass1 < imagesPerSet:
            cv2.imwrite(outputDirectory+'\\'+sets[0]+'\\'+classes[2]+'\\'+ self.imageName, self.currentImage)

        if imagesPerSet <= self.imageCounterClass1 < (2*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[1]+'\\'+classes[2]+'\\'+ self.imageName, self.currentImage)

        if (2*imagesPerSet) <= self.imageCounterClass1 < (3*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[2]+'\\'+classes[2]+'\\'+ self.imageName, self.currentImage)

        if self.imageCounterClass1 > (3*imagesPerSet):
            print "INFO: Don't need more images for " + classes[2]

        # Test if end of image list reached
        if self.imageCounter == (len(imagesList) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(imagesList) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass3 = self.imageCounterClass3 + 1

            # Show next image from list
            self.displayImage()

    def class4Button_clicked(self):
        print 'Image ' + imagesList[self.imageCounter] + ' is image number ' + str(self.imageCounterClass4 + 1) + ' classified as ' + classes[3] + '.'
        self.currentImage = cv2.imread(imagesList[self.imageCounter], 0)  # 0=grey, 1=RGB, -1=unchanged

        if self.imageCounterClass1 < imagesPerSet:
            cv2.imwrite(outputDirectory+'\\'+sets[0]+'\\'+classes[3]+'\\'+ self.imageName, self.currentImage)

        if imagesPerSet <= self.imageCounterClass1 < (2*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[1]+'\\'+classes[3]+'\\'+ self.imageName, self.currentImage)

        if (2*imagesPerSet) <= self.imageCounterClass1 < (3*imagesPerSet):
            cv2.imwrite(outputDirectory + '\\' + sets[2]+'\\'+classes[3]+'\\'+ self.imageName, self.currentImage)

        if self.imageCounterClass1 > (3*imagesPerSet):
            print "INFO: Don't need more images for " + classes[3]

        # Test if end of image list reached
        if self.imageCounter == (len(imagesList) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(imagesList) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            self.imageCounterClass4 = self.imageCounterClass4 + 1

            # Show next image from list
            self.displayImage()

    def class5Button_clicked(self):
        print 'Image ' + imagesList[self.imageCounter] + ' was classified as ' + classes[4] + ' and saved in ' + excludePath + '.'
        self.currentImage = cv2.imread(imagesList[self.imageCounter], 0)  # 0=grey, 1=RGB, -1=unchanged

        # Save to separate folder, disable if desired
        cv2.imwrite(excludePath+'\\'+ self.imageName, self.currentImage)

        # Test if end of image list reached
        if self.imageCounter == (len(imagesList) - 1):
            print "WARNING: There are no more images to classify."
            self.imageDisplay.setPixmap(QtGui.QPixmap(QtGui.QPixmap('..\\img\\logo_endOfList.png').scaled(self.height, self.width, QtCore.Qt.KeepAspectRatio)))
            self.class1Button.setEnabled(False)
            self.class2Button.setEnabled(False)
            self.class3Button.setEnabled(False)
            self.class4Button.setEnabled(False)
            self.class5Button.setEnabled(False)
            ## Add printing of number of images per folder.

        if self.imageCounter < (len(imagesList) - 1):
            self.imageCounter = self.imageCounter + 1
            self.Counter.display(self.imageCounter)
            # Numbers of images in class 5 is not tracked

            # Show next image from list
            self.displayImage()

    def cancelButton_clicked(self):
        print 'WARNING: MorphoSphere Data Set Generator was terminated by the user.'
        self.close()

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