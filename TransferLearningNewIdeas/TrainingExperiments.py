from pathlib import Path
import pandas as pd
import pickle
from collections import defaultdict
from TransferLearningNewIdeas.DatasetBuilder import Dataset
from PowerClassification.Utils.NetworkTraining import DataLoaderModule
from PowerClassification.Utils.NetworkTraining import NetworkFactoryModule, NetworkTrainingModule
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
import seaborn as sns

def experiment(experiment_path,datapath,dataset_config, conditions,rep=2):
    if not experiment_path.exists():
        experiment_path.mkdir(parents=True)

    acc_df = pd.DataFrame(columns=['rep','condition','train_acc','val_acc','test_acc'])
    training_curves_dict = defaultdict(list)
    #load data
    dataset = Dataset(datapath)
    dataset.create_train_val_test(*dataset_config)
    dataset.normalize()
    train_x, train_y, val_x, val_y, test_x, test_y = dataset.get_dataset()
    for c in conditions:
        for r in range(rep):
            #Create model
            training_module = NetworkTrainingModule()
            factory_module = NetworkFactoryModule()
            lstm_model, _ = factory_module.bestLstmModel(*(train_x.shape[1], train_x.shape[2]))
            if c == 'adam':
                opt = Adam(learning_rate=0.001)
            elif c == 'sgd':
                opt = SGD(learning_rate=0.01)
            else:
                raise Exception("Wrong optimizer")
            lstm_model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['acc'])
            print("repetition {:} condition {:}".format(r,c))
            print('Optimizer: ')
            print(lstm_model.optimizer)
            #Train model
            history, model, earlyStopping = training_module.trainModelEarlyStop(lstm_model, train_x, train_y, val_x,
                                                                                val_y, test_x, test_y, epochs=200,
                                                                                verbose=0, min_epochs=20)
            #save results
            epoch_of_max_acc = earlyStopping.epochOfMaxValidation
            training_curves_dict[c].append({'train':earlyStopping.trainingAcc[:epoch_of_max_acc],
                                            'val':earlyStopping.validationAcc[:epoch_of_max_acc],
                                            'test':earlyStopping.validationAcc2[:epoch_of_max_acc]})
            new_acc = {'rep':r,'condition':c,'train_acc':earlyStopping.trainingAcc[epoch_of_max_acc-1],
                       'val_acc':earlyStopping.validationAcc[epoch_of_max_acc-1],'test_acc':earlyStopping.validationAcc2[epoch_of_max_acc-1]}
            acc_df = acc_df.append(new_acc, ignore_index=True)

            #debug
            #print("evaluate {:0.3f} curve {:0.3f}".format(lstm_model.evaluate(train_x, train_y, verbose=False)[1],earlyStopping.trainingAcc[epoch_of_max_acc-1]))
            print("val  evaluate {:0.3f} curve {:0.3f}".format(lstm_model.evaluate(val_x, val_y, verbose=False)[1],earlyStopping.validationAcc[epoch_of_max_acc-1]))
            print("test evaluate {:0.3f} curve {:0.3f}".format(lstm_model.evaluate(test_x, test_y, verbose=False)[1],earlyStopping.validationAcc2[epoch_of_max_acc-1]))
            #save temp
            acc_df.to_csv(experiment_path / 'temp_acc_df.csv')
            pickle.dump(training_curves_dict, open(experiment_path / 'temp_train_curves.pic', 'wb'))
    acc_df.to_csv(experiment_path/'acc_df.csv')
    pickle.dump(training_curves_dict,open(experiment_path/'train_curves.pic','wb'))

def show_train_curves(exp_dir):
    training_curves = pickle.load(open(exp_dir/'train_curves.pic','rb'))
    fig,axes = plt.subplots(3,1)
    for key, color in zip(training_curves.keys(),['r','b','g','o']):
        for r in range(len(training_curves[key])):
            axes[0].plot(training_curves[key][r]['train'],color,label=key)
            axes[1].plot(training_curves[key][r]['val'],  color, label=key)
            axes[2].plot(training_curves[key][r]['test'], color, label=key)
    legend_without_duplicate_labels(axes[0])
    plt.show()

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))
def show_box_plots(exp_dir):
    acc_df = pd.read_csv(exp_dir/"acc_df.csv",index_col=0)
    new_df = acc_df.melt(id_vars=["rep", "condition"], var_name="type", value_name="acc")
    sns.boxplot(x='type', y='acc', hue='condition', data=new_df)
    plt.show()
def do_experiments(exp_path):
    #Dataset configuration
    datapath = r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\feature-selection-pyprep-black-list\05s"
    train_u = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']; train_s=['0','1']
    val_u = train_u; val_s=['2']
    test_u = train_u; test_s= ['3']
    dataset_config1 = [train_u,val_u,test_u,train_s,val_s,test_s]
    experiment(exp_path,datapath,dataset_config1,['sgd','adam'],rep=5)

def show_experiment_results(exp_dir):
    show_train_curves(exp_dir)
    show_box_plots(exp_dir)

def main():
    #Do experiment
    exp_path = Path('./experiments/01_test')
    do_experiments(exp_path)
    #Show results of experiment
    exp_path = Path('./experiments/01_test')
    show_experiment_results(exp_path)

if __name__ == "__main__":
    main()