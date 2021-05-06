
from TransferLearningNewIdeas.DatasetBuilder import Dataset
from TransferLearningNewIdeas.DatasetBuilder import utils

def main():
    datapath = r"C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\data\feature-selection-pyprep-black-list\05s"
    train_u = ['UI01','UI02','UI03','UI04','UI05','UI06','UI07','UI08']; train_s=['0','1']
    val_u = train_u; val_s=['2']
    test_u = train_u; test_s= ['3']
    dataset_config1 = [train_u,val_u,test_u,train_s,val_s,test_s]

    m_name = "lstm_all_users"
    utils.main_train_model(datapath,dataset_config1,model_name=m_name)
    utils.main_test(datapath,dataset_config1,model_name=m_name)

if __name__ == "__main__":
   main()