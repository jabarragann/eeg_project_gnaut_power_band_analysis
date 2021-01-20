from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers  import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Softmax,LSTM, Bidirectional,TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import tensorflow.keras as keras
from tensorflow.keras import backend as K

class NetworkFactoryModule:
    @staticmethod
    def lstm2(timesteps, features):
        networkInput = Input(shape=(timesteps, features))

        dropout1 = Dropout(rate=0.5)(networkInput)
        hidden1 = Dense(4, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.5)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        hidden2 = Bidirectional(LSTM(4, stateful=False, dropout=0.5))(batchNorm1)
        hidden3 = Dense(2, activation='linear')(hidden2)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1, 'simpleModel'

    @staticmethod
    def createAdvanceLstmModel(timesteps, features, isBidirectional=True, inputLayerNeurons=64, inputLayerDropout=0.3,
                               lstmOutputSize=4):
        timesteps = timesteps
        features = features

        # Input layer
        networkInput = Input(shape=(timesteps, features))
        dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

        # First Hidden layer
        hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.5)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        # Choose if the network should be bidirectional
        if isBidirectional:
            lstmLayer = LSTM(lstmOutputSize, stateful=False,
                             dropout=0.5, kernel_regularizer=regularizers.l2(0.05))
            # hidden2 = Bidirectional( LSTM(lstmSize, stateful=False, dropout=0.5), merge_mode='concat' ) (batchNorm1)
            hidden2 = Bidirectional(lstmLayer, merge_mode='concat')(batchNorm1)
        else:
            hidden2 = LSTM(lstmOutputSize, stateful=False, dropout=0.5)(batchNorm1)

        hidden3 = Dense(2, activation='linear')(hidden2)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1, 'advanceModel'

    @staticmethod
    def hyperparameterTunning(timesteps, features, lstmLayers, lstmOutputSize,
                              isBidirectional, inputLayerNeurons=64, inputLayerDropout=0.3):

        timesteps = int(timesteps)
        features = int(features)
        lstmLayers = int(lstmLayers)
        lstmOutputSize =int(lstmOutputSize)
        isBidirectional =int(isBidirectional)

        # Input layer
        networkInput = Input(shape=(int(timesteps), int(features)))
        dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

        # First Hidden layer
        hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.5)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        out = batchNorm1
        for i in range(1, lstmLayers+1):
            retSeq = False if i == lstmLayers else True
            lstmLayer = LSTM(lstmOutputSize, stateful=False, return_sequences=retSeq,
                             dropout=0.5, kernel_regularizer=regularizers.l2(0.05))
            if isBidirectional:
                out = Bidirectional(lstmLayer, merge_mode='concat')(out)
            else:
                out = lstmLayer(out)

        hidden3 = Dense(2, activation='linear')(out)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1

    @staticmethod
    def experimentingWithLowerDropout(timesteps, features, lstmLayers, lstmOutputSize,
                              isBidirectional, inputLayerNeurons=64, inputLayerDropout=0.2):

        timesteps = int(timesteps)
        features = int(features)
        lstmLayers = int(lstmLayers)
        lstmOutputSize = int(lstmOutputSize)
        isBidirectional = int(isBidirectional)

        # Input layer
        networkInput = Input(shape=(int(timesteps), int(features)))
        dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

        # First Hidden layer
        hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.3)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        out = batchNorm1
        for i in range(1, lstmLayers + 1):
            retSeq = False if i == lstmLayers else True
            lstmLayer = LSTM(lstmOutputSize, stateful=False, return_sequences=retSeq,
                             dropout=0.3, kernel_regularizer=regularizers.l2(0.05))
            if isBidirectional:
                out = Bidirectional(lstmLayer, merge_mode='concat')(out)
            else:
                out = lstmLayer(out)

        hidden3 = Dense(2, activation='linear')(out)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1

    # Timesteps 7.0, features 128.0
    @staticmethod
    def bestLstmModel(timesteps, features, lstmLayers=2, lstmOutputSize=4.0,
                      isBidirectional=1.0, inputLayerNeurons=64, inputLayerDropout=0.3):

        timesteps = int(timesteps)
        features = int(features)
        lstmLayers = int(lstmLayers)
        lstmOutputSize =int(lstmOutputSize)
        isBidirectional =int(isBidirectional)

        # Input layer
        networkInput = Input(shape=(int(timesteps), int(features)))
        dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

        # First Hidden layer
        hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
        dropout2 = Dropout(rate=0.5)(hidden1)
        batchNorm1 = BatchNormalization()(dropout2)

        out = batchNorm1
        for i in range(1, lstmLayers+1):
            retSeq = False if i == lstmLayers else True
            lstmLayer = LSTM(lstmOutputSize, stateful=False, return_sequences=retSeq,
                             dropout=0.5, kernel_regularizer=regularizers.l2(0.05))
            if isBidirectional:
                out = Bidirectional(lstmLayer, merge_mode='concat')(out)
            else:
                out = lstmLayer(out)

        hidden3 = Dense(2, activation='linear')(out)
        networkOutput = Softmax()(hidden3)

        model1 = Model(inputs=networkInput, outputs=networkOutput)
        model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model1, 'bestModel'

    @staticmethod
    def EEGNet(nb_classes, Chans=32, Samples=128,
               dropoutRate=0.5, kernLength=int(64 / 2), F1=8,
               D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
        """ Keras Implementation of EEGNet
        http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

          nb_classes      : int, number of classes to classify
          Chans, Samples  : number of channels and time points in the EEG data
          dropoutRate     : dropout fraction
          kernLength      : length of temporal convolution in first layer. We found
                            that setting this to be half the sampling rate worked
                            well in practice. For the SMR dataset in particular
                            since the data was high-passed at 4Hz we used a kernel
                            length of 32.
          F1, F2          : number of temporal filters (F1) and number of pointwise
                            filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
          D               : number of spatial filters to learn within each temporal
                            convolution. Default: D = 2
          dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
        """
        K.set_image_data_format('channels_first')

        if dropoutType == 'SpatialDropout2D':
            dropoutType = SpatialDropout2D
        elif dropoutType == 'Dropout':
            dropoutType = Dropout
        else:
            raise ValueError('dropoutType must be one of SpatialDropout2D '
                             'or Dropout, passed as a string.')

        input1 = Input(shape=(1, Chans, Samples))

        ##################################################################
        block1 = Conv2D(F1, (1, kernLength), padding='same',
                        input_shape=(1, Chans, Samples),
                        use_bias=False)(input1)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                                 depth_multiplier=D,
                                 depthwise_constraint=max_norm(1.))(block1)
        block1 = BatchNormalization(axis=1)(block1)
        block1 = Activation('elu')(block1)
        block1 = AveragePooling2D((1, 4))(block1)
        block1 = dropoutType(dropoutRate)(block1)

        block2 = SeparableConv2D(F2, (1, 16),
                                 use_bias=False, padding='same')(block1)
        block2 = BatchNormalization(axis=1)(block2)
        block2 = Activation('elu')(block2)
        block2 = AveragePooling2D((1, 8))(block2)
        block2 = dropoutType(dropoutRate)(block2)

        flatten = Flatten(name='flatten')(block2)

        dense = Dense(nb_classes, name='dense',
                      kernel_constraint=max_norm(norm_rate))(flatten)
        softmax = Activation('softmax', name='softmax')(dense)

        model = Model(inputs=input1, outputs=softmax)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

        return model, 'EEGnet'

    @staticmethod
    def createConvLstmModel(input_shape, num_classes):
        networkInput = Input(shape=input_shape)

        #     model.add(TimeDistributed(Conv2D(8, (3, 3), padding='same')))
        # #     model.add(Dropout(rate=0.3))
        #     model.add(TimeDistributed(Activation('relu')))
        #     model.add(Dropout(rate=0.3))
        x = TimeDistributed(Conv2D(16, (3, 3)))(networkInput)
        x = TimeDistributed(Activation('relu'))(x)
        x = TimeDistributed(MaxPooling2D(pool_size=(2, 2))) (x)
        #     model.add(Dropout(0.25))

        x = TimeDistributed(Flatten())(x)
        x = TimeDistributed(Dense(45))(x)
        x = TimeDistributed(Activation('relu'))(x)
        # model.add(Dropout(0.5))

        # LSTM layer
        x = Bidirectional(LSTM(20, stateful=False, dropout=0.35))(x)

        x = Dense(num_classes)(x)
        networkOutput = Activation('softmax')(x)

        model = Model(inputs=networkInput, outputs=networkOutput)
        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        return model, "convolutional lstm"
