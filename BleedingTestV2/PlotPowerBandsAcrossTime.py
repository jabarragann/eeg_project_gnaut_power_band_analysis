from BleedingTestV2.Utils import *


def plot_channels_means(data, ch, band, color, ax):
    rolling_mean_blood = data.loc["Blood", (ch, band)].rolling(window=15).mean().fillna(0)
    rolling_mean_Noblood = data.loc["NoBlood", (ch, band)].rolling(window=15).mean().fillna(0)
    x = range(data.loc["Blood", (ch, band)].shape[0])
    ax.plot(x, rolling_mean_blood.values, '-', color=color, label="Blood" + ch)
    x = range(data.loc["NoBlood", (ch, band)].shape[0])
    ax.plot(x, rolling_mean_Noblood.values, '--', color=color, label="NoBlood" + ch)

    return ax

def main():
    mne.set_log_level("WARNING")
    path_dicts = {1:{"Blood"  : r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Juan\11-09-20\S01_T01_Bleeding\UJuan_S01_T01_VeinSutureBleeding_raw.txt",
                     "NoBlood": r"C:\Users\asus\OneDrive - purdue.edu\RealtimeProject\Experiments3-Data\VeinLigationSimulator-Tests\Juan\11-09-20\S01_T01_NoBleeding\UJuan_S01_T01_VeinSutureNoBleeding_raw.txt"}}
    path_dicts = path_dicts[1]

    start_times = {"Blood": 290.0, "NoBlood":375.0}

    results = []
    for k,p in path_dicts.items():

        path = Path(p)
        print(k,path.name,path.exists())

        raw = loadDataInRaw(path)

        epochs = splitDataIntoEpochs(raw,2.0,1.0)
        ts, powerbands = calculatePowerBand(epochs,'na',2,'na')

        #Crop to the specific segment
        seg = ts.loc[ (ts['ts']>start_times[k]) & (ts['ts']<start_times[k]+75)]
        powerbands = powerbands.loc[seg.index,:]

        #Concat
        results.append(pd.concat([powerbands], keys=[k], names=['Condition']))

    results = pd.concat(results)

    x=0

    fig,axes =plt.subplots(1,1)
    # ch, band = "FZ","Delta"
    # rolling_mean_blood = results.loc["Blood",  (ch,band)].rolling(window=15).mean().fillna(0)
    # rolling_mean_Noblood = results.loc["NoBlood",  (ch,band)].rolling(window=15).mean().fillna(0)
    # axes[0].plot(results.loc["Blood",  (ch,band)].values,label="Blood")
    # axes[0].plot(rolling_mean_blood.values,label="Blood")
    # axes[1].plot(results.loc["NoBlood",(ch,band)].values, label="NoBlood")
    # axes[1].plot(rolling_mean_Noblood.values, label="NoBlood")

    ch = "Delta"
    plot_channels_means(results,"FZ",ch,"blue", axes)
    # plot_channels_means(results,"T7",ch,"orange", axes[2])
    plot_channels_means(results,"T8",ch,"black",axes)

    axes.axvline(14)
    axes.legend()


    plt.show()


if __name__ == "__main__":
    main()