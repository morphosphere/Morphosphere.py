######### SEGMENT SPHEROID Debugging #########
    #inputImagePath = "Z:\\Vardan_Andriasyan\\Morphosphere\\HT-29\\Test\\NonSphHealthy\\20160126-corning-all-spheroids-p2-004hps_D03_w1.TIF"
    #dilationDisk = 10
    #blockSize = 501
    #minSpheroidArea = 10000
    #start = time.clock()

import skimage
import math
import cv2
from scipy import ndimage
from skimage import measure,morphology
import matplotlib.pyplot as plt
import numpy as np

#dead spheroid
#inputImagePath= 'N:\\Vardan_Andriasyan\\Morphosphere\\HT-29\\Test\\NonSphNonHealthy\\20160205-corning-all-spheroids-p4-242hps_G05_w1.TIF'
#healthy spheroid
inputImagePath= 'N:\\Vardan_Andriasyan\\Morphosphere\\HT-29\\Test\\SphHealthy\\20150721-HTSassay-HT29-3D-d3_A04_w3.TIF'
dilationDisk = 8
blockSize = 501
minSpheroidArea = 20000
plt.set_cmap('gray')

inputImage = cv2.imread(inputImagePath, -1) #0 = grey, 1=RGB, .. -1 = as is

processedImage = cv2.convertScaleAbs(inputImage, alpha=(255.0/65535.0)) #convert to unsigned 8bit
#processedImage = inputImage.astype('uint8')  #convert to unsigned 8bit
#processedImage = skimage.util.img_as_ubyte(inputImage) #
#processedImage = inputImage #don't convert, then doesn't work
# here error for conversion

plt.figure(1)
plt.imshow(inputImage)
plt.figure(2)
plt.imshow(processedImage)
plt.show()

thresholdedImage = cv2.adaptiveThreshold(processedImage,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,0) #return inverted threshold, 0 = threshold correction factor
plt.figure(3)
plt.imshow(thresholdedImage)
plt.show()

selem = skimage.morphology.disk(dilationDisk)

thresholdedImage = cv2.dilate(thresholdedImage,selem,iterations = 1)

thresholdedImage = ndimage.binary_fill_holes(thresholdedImage)

plt.figure(4)
plt.imshow(thresholdedImage)
plt.show()

labeledImage = measure.label(thresholdedImage)
allProperties = measure.regionprops(labeledImage)


imageHeight, imageWidth = thresholdedImage.shape[:2]
#initialize empty arrays for area and distance filtering
areas=np.empty(len(allProperties))
distancesToCenter = np.empty(len(allProperties))
labels = np.empty(len(allProperties))

#find the index connected area which is closest to center of the image also area filter
i = 0
for props in allProperties:
    y0, x0 = props.centroid
    distance = math.sqrt((y0 -imageHeight/2)**2 + (x0 -imageWidth/2)**2)
    distancesToCenter[i] = distance
    areas[i] = props.area
    labels[i] = props.label
    i=i+1

#filter by area
distancesToCenter = distancesToCenter[areas>minSpheroidArea]
labels = labels[areas>minSpheroidArea]
# filter by distance and get the index
indexOfMinDistance = labels[distancesToCenter == min(distancesToCenter)]
indexOfMinDistance = indexOfMinDistance.astype(int) -1

selectedCC = (labeledImage == allProperties[indexOfMinDistance].label)

spheroidBWImage = selectedCC.astype("uint8")

plt.figure(5)
plt.imshow(spheroidBWImage)
plt.show()

boundingBox =  allProperties[indexOfMinDistance].bbox

#get all geometric measurements
centroid = allProperties[indexOfMinDistance].centroid
perimeter = allProperties[indexOfMinDistance].perimeter
area = allProperties[indexOfMinDistance].area
diameter = allProperties[indexOfMinDistance].equivalent_diameter
majorAxis = allProperties[indexOfMinDistance].major_axis_length
minorAxis = allProperties[indexOfMinDistance].minor_axis_length
circularity = 4*math.pi*(area/perimeter**2)

# Make output Images square

outputImageSide = max(boundingBox[2]-boundingBox[0],boundingBox[3]-boundingBox[1])

minRow = round(centroid[0]-outputImageSide/2)
maxRow = round(centroid[0]+outputImageSide/2)
minCol = round(centroid[1]-outputImageSide/2)
maxCol = round(centroid[1]+outputImageSide/2)

croppedBWImage = spheroidBWImage[minRow:maxRow,minCol:maxCol]
croppedImage  =  inputImage[minRow:maxRow,minCol:maxCol]*croppedBWImage

spheroidAttributes = {'area': area ,'diameter': diameter,'circularity': circularity,'majorAxis': majorAxis,'minorAxis': minorAxis}

fullImage = inputImage*spheroidBWImage

#plt.figure(3)
#plt.imshow(croppedBWImage)
#plt.figure(4)
#plt.imshow(croppedImage)
#
## plot_comparison(inputImage,spheroidBWImage,inputImagePath)
#plt.show()






