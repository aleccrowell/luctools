import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt; plt.rcParams['image.cmap'] = 'Set1'
import seaborn as sns; sns.set(color_codes=True); sns.set(style="white", context="talk")
import math
import peakutils
from scipy.stats import vonmises


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
            if peak:
                self.periods.append(np.abs(self.srate/freq[peak])[0])
            else:
                self.periods.append(np.nan)

    def get_phases(self):
        self.phases = []
        for i in range(self.data.values.shape[1]):
            t = np.arange(len(self.data.values[:, i]))
            sp = np.fft.fft(self.data.values[:, i])
            freq = np.fft.fftfreq(t.shape[-1])
            if not np.isnan(self.periods[i]):
                self.phases.append(self.periods[i]*(np.angle(sp[np.where(freq == self.srate/self.periods[i])])[0])/(2*np.pi))
            else:
                self.phases.append(np.nan)
        self.phases = [i % (2*np.pi) for i in self.phases]

#Just produce phase period plot for initial release
    def gen_phase_plot(self, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
        labels = np.array([i.split('.')[0] for i in self.data.columns.values[np.where(np.logical_and(~np.isnan(self.phases), ~np.isnan(self.periods)))]])
        phases = np.array(self.phases)[np.where(np.logical_and(~np.isnan(self.phases), ~np.isnan(self.periods)))]
        periods = np.array(self.periods)[np.where(np.logical_and(~np.isnan(self.phases), ~np.isnan(self.periods)))]
        for i in np.unique(labels):
            ax.scatter(phases[np.where(np.array(labels) == i)],periods[np.where(np.array(labels) == i)], label = i)
        #ax.set_title(filename, va='bottom')
        #plt.figtext(0.78, 0.2, 'p value of equal \nphases = '+str(round(p, 2)))
        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi/2)
        ax.set_thetagrids(angles=np.linspace(
            0, 360, 12), labels=range(0, 22, 2))
        ax.set_rlabel_position(0)
        #ax.set_rorigin(0)
        ax.set_rmax(max([(np.nanmax(self.periods)+1),24]))
        ax.set_rmin(min([(np.nanmin(self.periods)-4),18]))
        ax.set_rticks(range(int(min([(np.nanmin(self.periods)-1), 18])),int(max([(np.nanmax(self.periods)+1), 24])), 1))
        rlabels = result = [None]*(len(range(int(min([(np.nanmin(self.periods)-1), 18])), int(max([(np.nanmax(self.periods)+1), 24])), 1)))
        rlabels[::2] = np.arange(min([(int(np.nanmin(self.periods))-1), 18]), max([(int(np.nanmax(self.periods))+1), 24]), 2)
        ax.set_yticklabels(rlabels)
        plt.legend(bbox_to_anchor=[1.45, 1.1])
        plt.savefig(filename+'_phase_v_period.pdf')
        plt.close()

#For future development
    def compare_phases(self):
        #need to remember to subtract boundary cdf
        #once I find intersections, need to determine which pdf is higher inbetween each pair of adjacent intersections and take the cdf segment from the
        #  lower one
        #can test whether this would work by generating data from two gaussians, comparing pdf overlap and t test
        genotypes = np.array([i.split('.')[0] for i in self.data.columns.values])
        phases_1 = np.array(self.phases)[np.where(genotypes == np.unique(genotypes)[0])[0]]
        phases_1 = phases_1[~np.isnan(phases_1)]
        phases_2 = np.array(self.phases)[np.where(genotypes == np.unique(genotypes)[1])[0]]
        phases_2 = phases_2[~np.isnan(phases_2)]
        self.vm1 = vonmises.fit(phases_1, fscale=1)
        self.vm2 = vonmises.fit(phases_2, fscale=1)



