# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './../')
import SimpleLstmClassification as lstmClf

# Use scikit-learn to grid search the batch size and epochs
import numpy as np
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization,Softmax,LSTM, Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pickle

#Lstm mode to test
def createLstmModel(isBidirectional=True, inputLayerNeurons= 8, inputLayerDropout=0.5, lstmSize = 4):
    timesteps = 12
    features = 150

    #Input layer
    networkInput = Input(shape=(timesteps, features))
    dropout1 = Dropout(rate=inputLayerDropout)(networkInput)

    #First Hidden layer
    hidden1 = Dense(inputLayerNeurons, activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.5)(hidden1)
    batchNorm1 = BatchNormalization()(dropout2)

    #Choose if the network should be bidirectional
    if isBidirectional:
        hidden2 = Bidirectional( LSTM(lstmSize, stateful=False, dropout=0.5), merge_mode='concat' ) (batchNorm1)
    else:
        hidden2 = LSTM(lstmSize, stateful=False, dropout=0.5) (batchNorm1)

    hidden3 = Dense(2, activation='linear')(hidden2)
    networkOutput = Softmax()(hidden3)

    model1 = Model(inputs=networkInput, outputs=networkOutput)
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model1

def getData():
    trainX = []
    trainY = []
    trainingUsers = ['ryan', 'jhony', 'jackie', 'juan']

    for u in trainingUsers:
        p = './../data/users/{:}/'.format(u)
        dataContainer = lstmClf.getDataSplitBySession(p)

        availableSessions = np.array(list(dataContainer.keys()))
        trainX.append(np.concatenate([dataContainer[i]['X'] for i in availableSessions]))
        trainY.append(np.concatenate([dataContainer[i]['y'] for i in availableSessions]))

    trainX = np.concatenate(trainX)
    trainY = np.concatenate(trainY)

    return trainX, trainY


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Load data
X,Y = getData()

# create model
model = KerasClassifier(build_fn=createLstmModel, epochs=100, batch_size=256, verbose=1)

# define the grid search parameters
isBidirec = [True, False]
param_grid = dict(inputLayerNeurons= [8, 32, 64], inputLayerDropout=[0.1, 0.3, 0.5], lstmSize = [4, 14, 24])

#Execute the GridSearch
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=2)

grid_result = grid.fit(X, Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

f = open('./gridResult.pickle','wb')
results = {'cv_results':grid_result.cv_results_, 'best_params':grid_result.best_params_,'best_score_' :grid_result.best_score_}
pickle.dump(results, f)
f.close()