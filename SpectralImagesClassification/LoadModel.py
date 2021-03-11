from tensorflow.keras.models import load_model
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed, Input, BatchNormalization, Softmax
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model


def createLstmModel(input_shape, num_classes, lstmLayers=1, lstmOutputSize=4.0,
                    isBidirectional=0.0, inputLayerNeurons=10, inputLayerDropout=0.55):
    dropoutRate = inputLayerDropout

    lstmLayers = int(lstmLayers)
    lstmOutputSize = int(lstmOutputSize)
    isBidirectional = int(isBidirectional)

    # Input layer
    networkInput = Input(shape=input_shape)
    dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

    # First Hidden layer
    hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
    dropout2 = Dropout(rate=dropoutRate)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    out = batchNorm1
    for i in range(1, lstmLayers + 1):
        retSeq = False if i == lstmLayers else True
        lstmLayer = LSTM(lstmOutputSize, stateful=False, return_sequences=retSeq,
                         dropout=dropoutRate, kernel_regularizer=regularizers.l2(1.0))
        if isBidirectional:
            out = Bidirectional(lstmLayer, merge_mode='concat')(out)
        else:
            out = lstmLayer(out)

    hidden3 = Dense(num_classes, activation='linear')(out)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model1


class simple_lstm_predictor:
    def __init__(self, lstm_model_name):
        #Load model and configuration
        self.model = load_model('./models/{:}.h5'.format(lstm_model_name))
        self.normalization_dict = pickle.load(open('./models/{:}_normalizer.pickle'.format(lstm_model_name), 'rb'))
        self.configuration_dict = pickle.load(open('./models/{:}_config.pickle'.format(lstm_model_name), 'rb'))

        self.sf =250
        self.window_length = self.configuration_dict['frame_length']
        self.overlap = self.configuration_dict['overlap']
        self.lstm_sequence_length = self.configuration_dict['sequence_length']

        self.window_size = int(self.sf*self.window_length)
        self.chunk_size  = int(self.sf*self.window_length - self.sf*self.overlap)

        self.dataBuffer = np.zeros((30000,30))
        self.sequenceForPrediction = np.zeros((1, self.lstm_sequence_length, 90))

        #Load prediction model and normalizer
        self.global_mean = self.normalization_dict['mean']
        self.global_std = self.normalization_dict['std']

new_model = simple_lstm_predictor("simple_lstm_seq20_keyu-peg-vs-knot-test-4-6")

# sequence_length = 20
# num_classes = 2
# input_shape = (sequence_length,90)
# model = createLstmModel(input_shape, num_classes, inputLayerDropout=0.50)
# model.load_weights('./models/simple_lstm_seq20_keyu-peg-vs-knot-test-4-6_weights.h5')

new_model.summary()