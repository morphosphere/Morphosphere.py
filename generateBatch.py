# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:06:37 2016

@author: Vardan
"""


def generateBatch(currentSpheroidSet, currentLabelSet, maximumImageSize = 1358, outputImageSize = 32, numberOfReplicates = 11):

#transform and expand data for deep learning

    #segmentationData = pickle.load(open("segmentationData.p", "rb"))
    #currentSpheroidSet = segmentationData['testSpheroids']
    #currentLabelSet = segmentationData['testLabels']
    # import cPickle as pickle

    import numpy as np
    #import matplotlib.pyplot as plt

    import skimage
    from skimage import transform
    import sklearn

    def randomtransform(inputImage):
        outputImage = inputImage
        # random flip
        randomNum = np.random.randint(2)
        if randomNum == 0:
            outputImage = np.flipud(outputImage)
        if randomNum == 1:
            outputImage = np.fliplr(outputImage)
        randomNum = np.random.randint(181)
        outputImage = skimage.transform.rotate(outputImage, randomNum)
        return outputImage
    #test for randomTransform
    '''
    outputImage = randomtransform(segmentationData['testSpheroids'][1])
    plt.figure(1)
    plt.imshow(segmentationData['testSpheroids'][1])
    plt.figure(2)
    plt.imshow(outputImage)
    plt.set_cmap('gray')
    plt.show()

    '''
    i=0
    iLabel = 0
    for currentImage in currentSpheroidSet:
        imageHeight, imageWidth = currentImage.shape[:2]
        
        for iRep in range(0,numberOfReplicates):
            canvas = np.zeros((maximumImageSize, maximumImageSize), np.uint8)
            if iRep == 0:
                if imageHeight < maximumImageSize:
                    minRow = int(round(maximumImageSize/2 - imageHeight/2))
                    maxRow = int(round(maximumImageSize/2 + imageHeight/2))
                    minCol = int(round(maximumImageSize/2 - imageHeight/2))
                    maxCol = int(round(maximumImageSize/2 + imageHeight/2))
                    canvas[minRow:maxRow,minCol:maxCol] = currentImage
                else:
                    canvas = currentImage
            else:
                canvas = randomtransform(canvas)

            currentRescaledImage = skimage.transform.resize(canvas,(outputImageSize,outputImageSize)) # rescale the canvas    
            if i==0:
                currentBatchSpheroids = np.reshape(currentRescaledImage,(1,outputImageSize*outputImageSize))

                currentBatchLabels = currentLabelSet[iLabel]
            else:
                currentTransformedImage = np.reshape(currentRescaledImage, (1, outputImageSize * outputImageSize))
                currentBatchSpheroids = np.vstack((currentBatchSpheroids,currentTransformedImage))
                currentBatchLabels = np.vstack((currentBatchLabels,currentLabelSet[iLabel]))
            i += 1
        iLabel += 1



    #randomize Rows
    currentBatchSpheroids, currentBatchLabels = sklearn.utils.shuffle(currentBatchSpheroids, currentBatchLabels)

    bitDepth =255.0
    currentBatchSpheroids = currentBatchSpheroids.astype(np.float32) / bitDepth
    currentBatchLabels = np.squeeze(currentBatchLabels.astype(np.int64))
    outputTuple = (currentBatchSpheroids,currentBatchLabels)
    return outputTuple
    '''
    plt.imshow(currentBatchSpheroids)

    plt.set_cmap('gray')
    plt.show()
    '''


