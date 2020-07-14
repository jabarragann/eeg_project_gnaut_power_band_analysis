import pickle
import numpy as np
from collections import defaultdict
import re
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers  import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
import tensorflow.keras as keras

class NetworkFactoryModule:
    @staticmethod
    def EEGNet(nb_classes, Chans=32, Samples=128,
               dropoutRate=0.5, kernLength=64/2, F1=8,
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

        return Model(inputs=input1, outputs=softmax)


class DataLoaderModule:
    '''
        Utility to train data.
    '''
    # Channels PO7 and PO8 are not included
    EEG_CHANNELS = [
        "FP1", "FP2", "AF3", "AF4", "F7", "F3", "FZ", "F4",
        "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "CZ",
        "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
        "PZ", "P4", "P8", "PO3", "PO4", "OZ"]

    def getDataSplitBySession(self, datapath, debug = True):
        """
        Get a dictionary of all the different EEG in time format sessions found in the datapath. A session is defined as all the data
        that was collected before the person takes of the sensor.

        :param datapath:
        :param debug:
        :return:
        """

        # # Get a list of all the files
        datapath = Path(datapath)

        # Create container
        dataContainer = defaultdict(lambda: defaultdict(list))

        # Read all data files and concatenate them into a single dataframe.
        for f in datapath.rglob("*.pickle"):
            # Get Session
            sessId = re.findall("S[0-9]_T[0-9]", str(f.name))[0][-4]

            # Get data frame
            print(f) if debug else None

            with open(f,'rb') as fHandler:
                data = pickle.load(fHandler)
                assert isinstance(data, dict), "Unpickled data file is not a dict"
                assert 'X' in data.keys() and 'y' in data.keys(), "Unpickled eeg data not in the right format. Missing 'X' or 'y' key"


            # Append trial to dataset
            dataContainer[sessId]['X'].append(data['X'])
            dataContainer[sessId]['y'].append(data['y'])

        for key, value in dataContainer.items():
            dataContainer[key]['X'] = np.concatenate(dataContainer[key]['X'])
            dataContainer[key]['y'] = np.concatenate(dataContainer[key]['y'])

        return dataContainer


if __name__ == '__main__':

    dataLoader = DataLoaderModule()

    data = dataLoader.getDataSplitBySession("./../data/DifferentWindowSizeData_pyprep/10s",debug=True)
    x=0