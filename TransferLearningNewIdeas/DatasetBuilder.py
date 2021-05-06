from pathlib import Path
from tensorflow.keras.utils import to_categorical
from PowerClassification.Utils.NetworkTraining import DataLoaderModule
from PowerClassification.Utils.NetworkTraining import NetworkFactoryModule, NetworkTrainingModule
import numpy as np
from tensorflow.keras.models import load_model
import pickle

class Dataset:
    POWER_COEFFICIENTS = ['Delta', 'Theta', 'Alpha', 'Beta']
    def __init__(self, datapath):
        self.datapath = Path(datapath)
        self.users = []
        self.dataset_dict = {}
        self.data_loader = DataLoaderModule(dataFormat='freq') #lstm steps
        #Load data into dataset dict
        self.load_data()

        #Train-val-test data
        self.train_x = []; self.train_y = []
        self.val_x = []; self.val_y = []
        self.test_x = []; self.test_y = []
        #List containing the user id of each sample
        self.train_user_id = []
        self.val_user_id = []
        self.test_user_id = []
        #Normalizers
        self.X_mean=0
        self.X_std =1.0
    def load_data(self):
        #Load the subdirectories containing data from each user
        subdirectories = [x for x in self.datapath.iterdir() if x.is_dir()]

        for p in subdirectories:
            print("Loading user: ",p.name)
            data_container = self.data_loader.getDataSplitBySession(p,timesteps=int(80/5),powerBands=self.POWER_COEFFICIENTS, debug=False)
            self.dataset_dict[p.name]=dict(data_container)
    def normalize(self):
        # Convert labels to one-hot encoding
        self.train_y = to_categorical(self.train_y)
        self.val_y = to_categorical(self.val_y)
        self.test_y = to_categorical(self.test_y)

        # Normalize data
        self.X_mean = np.mean(self.train_x, axis=(0, 1))
        self.X_std = np.std(self.train_x, axis=(0, 1))

        self.train_x = (self.train_x- self.X_mean) / (self.X_std + 1e-18)
        self.val_x = (self.val_x - self.X_mean) / (self.X_std + 1e-18)
        self.test_x = (self.test_x - self.X_mean) / (self.X_std + 1e-18)

    def normalize_from_dict(self, normalizer):
        # Convert labels to one-hot encoding
        self.train_y = to_categorical(self.train_y)
        self.val_y = to_categorical(self.val_y)
        self.test_y = to_categorical(self.test_y)

        # Normalize data
        self.X_mean = normalizer['mean']
        self.X_std = normalizer['std']

        self.train_x = (self.train_x - self.X_mean) / (self.X_std + 1e-18)
        self.val_x = (self.val_x - self.X_mean) / (self.X_std + 1e-18)
        self.test_x = (self.test_x - self.X_mean) / (self.X_std + 1e-18)

    def create_train_val_test(self, train_u,val_u,test_u,train_s,val_s,test_s):

        for u in train_u:
            keys = list(self.dataset_dict[u].keys())
            for s in train_s:
                s_2 = keys[int(s)]
                self.train_x.append(self.dataset_dict[u][s_2]['X'])
                self.train_y.append(self.dataset_dict[u][s_2]['y'])
                user_id_list =np.array([u]*self.dataset_dict[u][s_2]['y'].shape[0])
                self.train_user_id.append(user_id_list)
        for u in val_u:
            keys = list(self.dataset_dict[u].keys())
            for s in val_s:
                s_2 = keys[int(s)]
                self.val_x.append(self.dataset_dict[u][s_2]['X'])
                self.val_y.append(self.dataset_dict[u][s_2]['y'])
                user_id_list = np.array([u] * self.dataset_dict[u][s_2]['y'].shape[0])
                self.val_user_id.append(user_id_list)
        for u in test_u:
            keys = list(self.dataset_dict[u].keys())
            for s in test_s:
                s_2 = keys[int(s)]
                self.test_x.append(self.dataset_dict[u][s_2]['X'])
                self.test_y.append(self.dataset_dict[u][s_2]['y'])
                user_id_list = np.array([u] * self.dataset_dict[u][s_2]['y'].shape[0])
                self.test_user_id.append(user_id_list)

        self.train_x = np.concatenate(self.train_x)
        self.train_y = np.concatenate(self.train_y)
        self.val_x = np.concatenate(self.val_x)
        self.val_y = np.concatenate(self.val_y)
        self.test_x = np.concatenate(self.test_x)
        self.test_y = np.concatenate(self.test_y)
        self.train_user_id = np.concatenate(self.train_user_id)
        self.val_user_id = np.concatenate(self.val_user_id)
        self.test_user_id = np.concatenate(self.test_user_id)

    def get_dataset(self):
        return self.train_x,self.train_y,self.val_x, self.val_y,self.test_x, self.test_y


class utils:
    @staticmethod
    def calculate_acc_per_user(model,data_x,data_y,data_user_id):
        for u in list(np.unique(data_user_id)):
            idx = np.where(data_user_id==u)
            y = data_y[idx]
            x = data_x[idx]
            results = model.evaluate(x,y,verbose=False)
            print("{:} acc {:0.3f}".format(u,results[1]))

    @staticmethod
    def main_train_model(datapath,dataset_config,model_name):
        dataset = Dataset(datapath)
        dataset.create_train_val_test(*dataset_config)
        dataset.normalize()
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_dataset()
        x=0
        #Create model
        training_module = NetworkTrainingModule()
        factory_module = NetworkFactoryModule()
        lstm_model,_ = factory_module.bestLstmModel(*(train_x.shape[1], train_x.shape[2]))
        #Train model
        history, model, earlyStopping = training_module.trainModelEarlyStop(lstm_model,train_x,train_y,val_x,val_y,test_x,test_y,epochs=200, verbose=1,min_epochs=20)
        training_module.createPlot(history, ".", ".", earlyStopCallBack=earlyStopping,save=False,show_plot=True)
        #Save model
        lstm_model.save("./models/{:}.h5".format(model_name))
        #Save normalizers
        normalizer = {'mean':dataset.X_mean,'std':dataset.X_std}
        pickle.dump(normalizer,open('./models/normalizer_{:}.pickle'.format(model_name),'wb'))
        print("Max validation accuracy epoch {:d}".format(earlyStopping.epochOfMaxValidation))
        print("Max validation accuracy {:0.3f}".format(earlyStopping.maxValidationAcc))

    @staticmethod
    def main_test(datapath,dataset_config,model_name):
        dataset = Dataset(datapath)
        dataset.create_train_val_test(*dataset_config)
        dataset.normalize()
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_dataset()

        lstm_model = load_model("./models/{:}.h5".format(model_name))
        results = lstm_model.evaluate(val_x,val_y,verbose=False)
        print("Val results", results[1])
        results = lstm_model.evaluate(test_x,test_y,verbose=False)
        print("test results", results[1])
        utils.calculate_acc_per_user(lstm_model,test_x,test_y,dataset.test_user_id)

    @classmethod
    def main_transfer(cls, datapath, dataset_config, model_name):
        dataset = Dataset(datapath)
        lstm_model = load_model("./models/{:}.h5".format(model_name))
        normalizer = pickle.load(open("./models/normalizer_{:}.pickle".format(model_name),'rb'))
        dataset.create_train_val_test(*dataset_config)
        dataset.normalize_from_dict(normalizer)
        train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_dataset()

        # Train model
        training_module = NetworkTrainingModule()
        history, model, earlyStopping = training_module.trainModelEarlyStop(lstm_model, train_x, train_y, val_x, val_y,
                                                                            test_x, test_y, epochs=200, verbose=1)
        training_module.createPlot(history, ".", ".", earlyStopCallBack=earlyStopping, save=False, show_plot=True)
        # Test module
        print("Max validation accuracy epoch {:d}".format(earlyStopping.epochOfMaxValidation))
        print("Max validation accuracy {:0.3f}".format(earlyStopping.maxValidationAcc))

        results = lstm_model.evaluate(val_x, val_y, verbose=False)
        print("Val results", results[1])
        results = lstm_model.evaluate(test_x, test_y, verbose=False)
        print("test results", results[1])
        cls.calculate_acc_per_user(lstm_model, test_x, test_y, dataset.test_user_id)
