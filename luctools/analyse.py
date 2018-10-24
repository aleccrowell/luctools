import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt; plt.rcParams['image.cmap'] = 'Set1'
import seaborn as sns; sns.set(color_codes=True); sns.set(style="white", context="talk")
import math
import peakutils


class luctraces:
    def __init__(self, filename, srate):
        self.data = pd.read_csv(filename, index_col=0)
        self.srate = srate
        self.notdone = True

    def gen_tsplot(self, fname):
        ldf = pd.melt(self.data.reset_index(), id_vars=[self.data.reset_index().columns.values.tolist()[0]], value_vars=self.data.columns.values.tolist())
        ldf['replicate'] = ldf['variable'].apply(lambda x: x.split('.')[1] if len(x.split('.')) != 1 else '0')
        ldf['variable'] = ldf['variable'].apply(lambda x: x.split('.')[0])
        ldf.columns = ['Frame', 'Genotype', 'Luminescence', 'Replicate']
        ldf['Time (hrs)'] = ldf['Frame']/self.srate
        fig = plt.figure()
        ax = plt.subplot(111)
        colors = ["windows blue", "amber", "greyish", "magenta", "dusty purple"]
        sns.lineplot(x="Time (hrs)", y="Luminescence", hue="Genotype", data=ldf, color=sns.xkcd_palette(colors), ci=95, ax=ax)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        for i in range((int(np.floor(ldf['Time (hrs)'].min())) + 24), int(np.floor(ldf['Time (hrs)'].max())), 24):
            ax.axvline(x=i, color='k', linestyle='--')
        plt.xticks(range(0, int(np.floor(ldf['Time (hrs)'].max())), 24))
        ax.set_xlim(int(np.floor(ldf['Time (hrs)'].min())), int(np.floor(ldf['Time (hrs)'].max())))
        plt.savefig(fname+'.pdf', bbox_inches='tight')
        plt.close()

    def detrend(self):
        for i in range(len(self.data.columns.values)):
            self.data.iloc[:, i] = self.data.iloc[:, i].values - np.min(self.data.iloc[:, i].values) + 1
            a, c = np.polyfit(np.log(self.data.iloc[:, i].index + 1), self.data.iloc[:, i], 1)
            self.data.iloc[:, i] = self.data.iloc[:, i] - (a * np.log(self.data.iloc[:, i].index + 1) + c)

#First determine frequency by generating autocorrelation plot and then taking fft of that

#Then take fft of original data and get phase of corresponding frequency component


#to compare phase distributions
#need to fit scipy von mises distributions and get overlap of pdfs using cdfs
#for normal, this would be:
#area = norm.cdf(r,m2,std2) + (1.-norm.cdf(r,m1,std1))
#where r is the point of intersection of the pdfs and m and std are parameters of the distributions

    def get_autocorrs(self):
        self.autocorrs = []
        for i in range(self.data.values.shape[1]):
            temp = []
            for j in range(len(self.data.values[:, i])):
                temp.append(np.corrcoef(self.data.values[:, i], np.roll(self.data.values[:, i], j))[0,1])
            self.autocorrs.append(temp)


    def get_periods(self):
        self.periods = []
        for i in range(len(self.autocorrs)):
            t = np.arange(len(self.autocorrs[i]))
            sp=np.fft.fft(self.autocorrs[i])[3:-3]
            freq = np.fft.fftfreq(t.shape[-1])[3:-3]
            peak = peakutils.indexes(sp, thres=0.5, min_dist=len(self.autocorrs[i]))
            self.periods.append(np.abs(self.srate/freq[peak]))

    def get_phases(self):
        self.phases = []
        for i in range(self.data.values.shape[1]):
            t = np.arange(len(self.data.values.shape[i]))

