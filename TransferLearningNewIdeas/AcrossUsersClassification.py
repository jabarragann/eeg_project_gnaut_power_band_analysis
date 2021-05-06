from TransferLearningNewIdeas.DatasetBuilder import Dataset
from TransferLearningNewIdeas.DatasetBuilder import utils

def test_dataset_class(datapath,dataset_conf1,dataset_conf2):
    dataset = Dataset(datapath)
    dataset.create_train_val_test(*dataset_conf1)
    #dataset.normalize()
    train_x, train_y, val_x, val_y, test_x_1, test_y_1 = dataset.get_dataset()

    dataset = Dataset(datapath)
    dataset.create_train_val_test(*dataset_conf2)
    #dataset.normalize()
    train_x, train_y, val_x, val_y, test_x_2, test_y_2 = dataset.get_dataset()

def main():
    datapath = r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\feature-selection-pyprep-black-list\05s"
    train_u = ['UI01','UI02','UI03','UI04','UI05','UI06','UI08']; train_s=['0','1','2']
    val_u = train_u; val_s=['3']
    test_u = ['UI07']; test_s= ['2','3']
    transfer_u = ['UI07']; transfer_s= ['0']
    transfer_val_u = ['UI07']; transfer_val_s= ['1']
    dataset_config1 = [train_u,val_u,test_u,train_s,val_s,test_s]
    dataset_config2 = [transfer_u,transfer_val_u,test_u,transfer_s,transfer_val_s,test_s]

    #Idea 1 multi-user model
    #m_name = "lstm_base"
    #main_train_model(datapath,dataset_config1,model_name=m_name)
    #main_test(datapath,dataset_config1,model_name=m_name)

    #Idea 2 train only with transfer data
    m_name = "lstm_little"
    utils.main_train_model(datapath,dataset_config2,model_name=m_name)
    utils.main_test(datapath, dataset_config2, model_name=m_name)

    #Idea 3 transfer learning
    #m_name = "lstm_base"
    #utils.main_transfer(datapath,dataset_config2,m_name)


    # print(dataset_config1)
    # print(dataset_config2)

if __name__ == "__main__":
    main()
