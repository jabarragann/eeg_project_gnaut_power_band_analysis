from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

userNameMapping  = {'jackie':'Subject 1','ryan':'Subject 2', 'juan':'Subject 3',
                    'jhony' :'Subject 4','karuna':'Subject 5','santy':'Subject 6'}

def formatDataForAnalysis(path):
    df = pd.read_csv(path)

    # Format data - TestAccBefore as proportionOfTransfer =0
    users = df.User.unique()
    compilation = []
    for u in users:
        partOfData = df.loc[df['User'] == u].reset_index(drop=True)
        testSessions = partOfData.TestSession.unique()
        for sess in testSessions:
            tinyPart = partOfData.loc[partOfData['TestSession'] == sess].reset_index(drop=True)
            testAccBefore = tinyPart.loc[0, 'TestAccBefore']
            newRow = {'User': u, 'TestSession': sess, 'proportionOfTransfer': 0.0,
                      'TestAccAfter': testAccBefore}
            tinyPart = tinyPart.append(newRow, ignore_index=True)
            compilation.append(tinyPart)

    return pd.concat(compilation)

def createBrokenAxisPlot():
    fig2, (ax1, ax2)  = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [5, 1]})
    # Broken Axis formating
    # hide the spines between ax and ax2
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    # Change space between subplots
    subplotDict = {'top': 0.88, 'bottom': 0.135, 'left': 0.105, 'right': 0.955, 'hspace': 0.055, 'wspace': 0.2}
    fig2.subplots_adjust(**subplotDict)

    # Draw diagonal lines in the axes coordinates
    d = .008  # how big to make the diagonal lines in axes coordinates
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)  # arguments to pass to plot
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d * 6, 1 + d * 6), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d * 6, 1 + d * 6), **kwargs)  # bottom-right diagonal

    return fig2, (ax1, ax2)

if __name__ == '__main__':

    # Load results data
    timePath = Path('C:\\Users\\asus\\PycharmProjects\\eeg_project_gnaut_power_band_analysis\\TimeClassification\\results\\results_transfer10\\aa11_pyprep\\window10s_sampleSize140s.csv')
    freqPath = Path(r'C:\Users\asus\PycharmProjects\eeg_project_gnaut_power_band_analysis\PowerClassification\results\EegResults\results_transfer10\aa13a_pyprep_complete\window10s_sampleSize110s.csv')
    assert timePath.exists(), "Make sure time path is correctly set"
    assert freqPath.exists(), "Make sure freq path are correctly set"
    timeData = formatDataForAnalysis(timePath)
    timeData['Model Type'] = "Time Model"
    freqData = formatDataForAnalysis(freqPath)
    freqData['Model Type'] = "Frequency Model"

    combined = [freqData, timeData]
    combined = pd.concat(combined)

    fig2, (ax1, ax2) = createBrokenAxisPlot()

    sns.lineplot(data=combined, x="proportionOfTransfer", y="TestAccAfter", hue="Model Type",
                 style="Model Type", markers=True, dashes=False, ci=95, ax = ax1)
    legend = ax1.legend(fancybox=True, shadow=True, )

    ax1.set_xticks(combined["proportionOfTransfer"].unique())
    ax1.set_xticklabels(["{:0.1f}%".format(l*100) for l in ax1.get_xticks()])
    ax1.set_ylabel("")
    # zoom-in / limit the view to different portions of the data


    ax2.set_yticks([0, 0.10])
    ax1.set_yticks([1*i/10 for i in range(11)])
    ax1.set_yticks([1*i/20 for i in range(21)], minor=True)
    ax2.set_yticks([1 * i / 20 for i in range(21)], minor=True)
    ax1.set_ylim([0.48, 1.0])  # outliers only
    ax2.set_ylim([0, 0.12])

    # Title
    ax1.set_title("95% CI mean testing accuracy")
    # Common Y label
    fig2.text(0.03, 0.5, 'Mean testing Accuracy', va='center', rotation='vertical')
    # X label
    ax2.set_xlabel("% of calibration data")
    # Grid
    ax1.grid(which='both'); ax2.grid(which='both');
    ax1.set_facecolor('0.9'); ax2.set_facecolor('0.9')


    plt.show()
    x = 0