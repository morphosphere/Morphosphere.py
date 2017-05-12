import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import glob,os
import cv2

#theano
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import theano
import theano.tensor as T
import timeit
from segmentSpheroid import segmentSpheroid
from generateBatch import generateBatch


##############################################
#################4 CLASSES####################
##############################################
"""
Demo for MorphoSphere

@author: Vardan Andriasyan, Fanny Georgi
"""

#
parentDir = 'N:\\Fanny_Georgi\\1-12_TumorRemission\\20160126-1-12-4_Spheroid_Complete_Analysis'
dataTableClassifiedPath = parentDir +'\\'+ '20160512_HT29complete_strict_classified.csv'
sets = ['Training','Test','Validation']
classes = ['healthy_spheroid', 'healthy_non-spheroid', 'unhealthy_spheroid', 'dead_spheroid']

#inputImage= 'Z:\\Vardan_Andriasyan\\Morphosphere\\HT-29\\Test\\NonSphNonHealthy\\20160205-corning-all-spheroids-p4-242hps_G05_w1.TIF'
#fullImage, croppedImage, croppedBWImage, spheroidAttributes = segmentSpheroid(inputImage, 12, 501, 20000)

trainingSpheroids = []
testSpheroids = []
validationSpheroids = []
trainingLabels = []
testLabels = []
validationLabels = []

allSizes = []

# Load dataTableClassified
dataTableClassified = np.genfromtxt(dataTableClassifiedPath, dtype = object, delimiter =',', skip_header = 1)
dataTableClassified = dataTableClassified.astype(str)
dataTableClassified = np.char.strip(dataTableClassified)

# Define segmentation parameters
dilationDiskDefault = 14
dilationDisk0 = 16
blockSize = 501
minSpheroidArea = 20000
thresholdCorrectionDefault = 5
thresholdCorrection0 = 3

# Visualization flag for segmentation plots
visualize = False

for image in range(0,len(dataTableClassified)-1,1):
    if dataTableClassified[image, 15] == 'Training' or dataTableClassified[image, 15] ==  'Test' or dataTableClassified[image, 15] ==  'Validation':
        print dataTableClassified[image,0]
        # Segment clump using fine-tuned parameters
        if dataTableClassified[image,5] == '003' or dataTableClassified[image,5] == '004' or dataTableClassified[image,5] == '005' or dataTableClassified[image,5] == '006' or dataTableClassified[image,5] == '007':
            dilationDisk = dilationDisk0
            thresholdCorrection = thresholdCorrection0
        else:
            dilationDisk = dilationDiskDefault
            thresholdCorrection = thresholdCorrectionDefault
        #print 'Segmentation parameters are ' + str(dilationDisk) + str(thresholdCorrection
        croppedImage, croppedBWImage, spheroidAttributes = segmentSpheroid(dataTableClassified[image,0], dilationDisk, blockSize, minSpheroidArea, thresholdCorrection, visualize)
        # Populate an array with all image sizes to later obtain max size
        imageHeight, imageWidth = croppedImage.shape[:2]
        allSizes.append(max([imageHeight, imageWidth]))
        # Load cropped images into array structure
        if dataTableClassified[image, 15] == 'Training':
            trainingSpheroids.append(croppedImage)
            if dataTableClassified[image,14] == 'healthy_spheroid':
                trainingLabels.append([0])
            if dataTableClassified[image,14] == 'healthy_non-spheroid':
                trainingLabels.append([1])
            if dataTableClassified[image,14] == 'unhealthy_spheroid':
                trainingLabels.append([2])
            if dataTableClassified[image,14] == 'dead_spheroid':
                trainingLabels.append([3])
        if dataTableClassified[image,15] == 'Test':
            testSpheroids.append(croppedImage)
            if dataTableClassified[image,14] == 'healthy_spheroid':
                testLabels.append([0])
            if dataTableClassified[image,14] == 'healthy_non-spheroid':
                testLabels.append([1])
            if dataTableClassified[image,14] == 'unhealthy_spheroid':
                testLabels.append([2])
            if dataTableClassified[image,14] == 'dead_spheroid':
                testLabels.append([3])
        if dataTableClassified[image,15] == 'Validation':
            validationSpheroids.append(croppedImage)
            if dataTableClassified[image,14] == 'healthy_spheroid':
                validationLabels.append([0])
            if dataTableClassified[image,14] == 'healthy_non-spheroid':
                validationLabels.append([1])
            if dataTableClassified[image,14] == 'unhealthy_spheroid':
                validationLabels.append([2])
            if dataTableClassified[image,14] == 'dead_spheroid':
                validationLabels.append([3])

# Detect maximum image size
maximumImageSize = max(allSizes)
print 'Maximum image size is ' + str(max(allSizes)) + ' .'

# Load arrays of cropped images into dictionary
segmentationData = {"testSpheroids": testSpheroids, "testLabels": testLabels,"trainingSpheroids": trainingSpheroids, "trainingLabels": trainingLabels,"validationSpheroids": validationSpheroids, "validationLabels": validationLabels,"maximumImageSize": maximumImageSize}
# Pickle dictionary
pickle.dump(segmentationData, open( "segmentationData4Classes_HT29complete_strict.p", "wb" ) )

print 'Successfully created data sets for ' + str(dataTableClassifiedPath) + ' .'

##############################################################################################
# Load segmentation data, create random rotation replicates
##############################################################################################

from generateBatch import generateBatch

#segmentationData = pickle.load( open( "segmentationData2Classes.p", "rb" ) )
#numberOfClasses = 2

segmentationData = pickle.load( open("segmentationData4Classes_HT29complete_strict.p", "rb" ) )
numberOfClasses = 4

maximumImageSize = segmentationData["maximumImageSize"]

outputImageSize = 128
# Number of replicates boosting training set
numberOfReplicates = 10
trainingSet = generateBatch(segmentationData["trainingSpheroids"],segmentationData["trainingLabels"], maximumImageSize, outputImageSize, numberOfReplicates)
testingSet = generateBatch(segmentationData["testSpheroids"],segmentationData["testLabels"], maximumImageSize, outputImageSize, numberOfReplicates)
validationSet = generateBatch(segmentationData["validationSpheroids"],segmentationData["validationLabels"], maximumImageSize, outputImageSize, numberOfReplicates)

cnnData = {"testingSet": testingSet,"trainingSet": trainingSet,"validationSet": validationSet}
pickle.dump( cnnData, open( "cnnData_HT29complete_strict.p", "wb" ) )

print 'There are ' + str(len(trainingSet[0])) + 'training set images.'
print 'Successfully loaded the CNN data with for ' + str(dataTableClassifiedPath) + ' with ' + str(numberOfReplicates) + ' for each of the ' + str(len(trainingSet[0])) + ' training set images.'

##############################################################################################
# Load data sets
##############################################################################################

cnnData = pickle.load(open( "cnnData_HT29complete_strict.p", "rb" ) )
trainingSet = cnnData["trainingSet"]
testingSet = cnnData["testingSet"] 
validationSet = cnnData["validationSet"]

#numberOfClasses = 2
numberOfClasses = 4

outputImageSize = 128

print 'There are ' + str(len(trainingSet[0])) + ' training set images.'
batch_size = len(trainingSet[0])

print 'Successfully loaded data sets for ' + str(dataTableClassifiedPath) + '.'

##############################################################################################
# Convoltional Neural Network
##############################################################################################

from loadCNNData import loadCNNData
from CNNClasses import HiddenLayer,LogisticRegression,LeNetConvPoolLayer
import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv2d

validationError = []

# Function definition, contains default variable values
def evaluate_lenet5(learning_rate=0.1, n_epochs=200, nkerns=[20, 50], batch_size=400, numberOfClasses=2):

    rng = numpy.random.RandomState(23455)

    datasets = loadCNNData(trainingSet, testingSet, validationSet)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 128 * 128)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (128, 128) is the size of images.
    layer0_input = x.reshape((batch_size, 1, 128, 128))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
    # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 128, 128),
        filter_shape=(nkerns[0], 1, 37, 37),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
    # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 46, 46),
        filter_shape=(nkerns[1], nkerns[0], 7, 7),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        # n_in = nkerns[1] * (imageShapeLayer2 - filterSize +1)/2 * (imageShapeLayer2 - filterSize +1)/2
        n_in=nkerns[1] * 20 * 20,
        n_out=batch_size,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=batch_size, n_out=numberOfClasses)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 1  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print('training @ iter = ', iter)
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                validationError.append(this_validation_loss * 100.)

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ' + ' ran for %.2fm' % ((end_time - start_time) / 60.)))
    
    return validationError

# Define optimized input parameters here
# batch_size = len(trainingSet[0]) # (number of images per training set) * (number of classes) * (number of replicates)
validationError = evaluate_lenet5(learning_rate=0.01, n_epochs=200, nkerns=[20, 50], batch_size=400, numberOfClasses=4)

def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
    
print "Learning finished."