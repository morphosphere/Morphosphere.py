import theano
import theano.tensor as T
import numpy

def loadCNNData(trainingSet,testingSet,validationSet):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables
        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')
    '''
    def trasnformMatlabInput(inputImagePath, labelInputPath):
        # inputLabelPath = 'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleLabels_test.txt';
        # inputImagePath = 'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleBatch_test32.tif';
        inputLabel = numpy.loadtxt(labelInputPath, unpack=True, delimiter=', ')
        inputLabel = inputLabel.astype(numpy.int64)

        inputImg = mpimg.imread(inputImagePath)
        inputImg = (inputImg.astype(numpy.float32) / 255.0)

        outputSet = (inputImg, inputLabel)
        return outputSet

    #############
    # LOAD DATA #
    #############

    train_set = trasnformMatlabInput(
        'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleBatch_train32.tif',
        'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleLabels_train.txt')
    valid_set = trasnformMatlabInput(
        'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleBatch_valid32.tif',
        'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleLabels_valid.txt')
    test_set = trasnformMatlabInput(
        'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleBatch_test32.tif',
        'Z:\\Vardan_Andriasyan\\Morphosphere\\TrainingSet\\SimpleSet\\spheroidSimpleLabels_test.txt')
    '''
    test_set_x, test_set_y = shared_dataset(testingSet)
    valid_set_x, valid_set_y = shared_dataset(validationSet)
    train_set_x, train_set_y = shared_dataset(trainingSet)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    print(rval)
    return rval
