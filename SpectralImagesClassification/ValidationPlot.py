import mne
from SpectralImagesClassification.LoadDataUtils import renameChannels, renamedChannels
from SpectralImagesClassification.LoadDataUtils import splitDataIntoEpochs, getBandPowerCoefficients
from pathlib import Path

if __name__ == "__main__":

    #Parameters
    image_size = 32
    frame_length = 1 #args.segment_length
    sequence_length = 25  # 20
    overlap = 0.5
    num_classes = 2

    dataPath = Path(r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\CalibrationProcedure-NeedlePasssingBlood\edf\JuanValidation")
    dataPath = dataPath / "UJuan_S01_T01_BloodValidation_raw.edf"

    file = dataPath
    print("Processing file {:}".format(file.name))
    raw = mne.io.read_raw_edf(file)
    mne.rename_channels(raw.info, renameChannels)
    #Remove bad channels
    raw = raw.pick(renamedChannels)
    # Filter data
    raw.load_data()
    raw.filter(0.5, 30)
    #Get epochs
    epochs = splitDataIntoEpochs(raw, frame_length, overlap)
    epochs_ts = epochs.events[:,0]/250
    bandpower = getBandPowerCoefficients(epochs)

    #Load model
    x=0
    #Predict

    #Plot predictions