import sys
import getopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['image.cmap'] = 'Set1'
import seaborn as sns; sns.set(color_codes=True); sns.set(style="white", context="talk")
import scipy as sp
import scipy.optimize
import inflect
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
circularlib=importr('circular')
statslib=importr('stats')
circular=robjects.r('circular')
wwt=robjects.r('watson.williams.test')
rlist=robjects.r('list')
specar=robjects.r('spec.ar')
specpgram=robjects.r('spec.pgram')


def get_app(e,srate,mode):
    sp = np.fft.fft(e)
    freq = np.fft.fftfreq(e.shape[-1])
    pindex =  np.where(sp.real==np.max(sp.real[np.abs(freq) > (srate/48)]))
    phase = np.angle(sp[pindex])[0]
    amp = (np.abs(sp.real[pindex])**2)[0]
    #per = np.abs(srate/freq[pindex])[0]
    if mode == 'yw':
        spec = specar(robjects.FloatVector(e), 1000, order = 15, method = "yule-walker", plot='FALSE')
        pindex =  np.where(np.asarray(spec[1])==np.max(np.asarray(spec[1])[np.asarray(spec[0]) > (srate/72)]))
        per = 1/np.asarray(spec[0])[np.max(pindex[0])]
    elif mode =='fft':
        spec = specpgram(robjects.FloatVector(e), plot='FALSE')
        pindex =  np.where(np.asarray(spec[1])==np.max(np.asarray(spec[1])[np.asarray(spec[0]) > (srate/72)]))
        per = 1/np.asarray(spec[0])[np.max(pindex[0])]
    return amp, phase, per

def fit_exp_linear(t, y, C=0):
    y = y - C
    y = np.log(y)
    K, A_log = np.polyfit(t, y, 1)
    A = np.exp(A_log)
    return A, K


def gen_tsplot(df,fname):
    ldf = pd.melt(df.reset_index(), id_vars=[df.reset_index().columns.values.tolist()[0]], value_vars=df.columns.values.tolist())
    ldf['replicate'] = ldf['variable'].apply(lambda x: x.split('.')[1] if len(x.split('.')) != 1 else '0')
    ldf['variable'] = ldf['variable'].apply(lambda x: x.split('.')[0])
    ldf.columns = ['Frame','Genotype','Luminescence','Replicate']
    fig = plt.figure()
    ax = plt.subplot(111)
    sns.tsplot(time="Frame", value="Luminescence", unit='Replicate', condition="Genotype",data=ldf,color="Set1",ci=95,ax=ax)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(fname+'.pdf')
    plt.close()


def detrend(df):
    ndf = df.copy()
    for i in range(len(ndf.columns.values)):
        ndf.iloc[:,i] = ndf.iloc[:,i].values - np.min(ndf.iloc[:,i].values) + 1
        A, K = fit_exp_linear(ndf.iloc[:,i].index,ndf.iloc[:,i].values)
        ndf.iloc[:,i] = ndf.iloc[:,i] -(A * np.exp(K * ndf.iloc[:,i].index))
    return ndf


def get_stats(df,sr=1,m='yw'):
    ndf = df.copy()
    newdict = {}
    for i in range(len(ndf.columns.values)):
        newdict[ndf.columns.values[i]] = get_app(ndf.iloc[:,i].values,sr,m)
    amps = {}
    phases = {}
    pers = {}
    for k, v in newdict.items():
        if k.split('.')[0] in amps.keys():
            amps[k.split('.')[0]].append(v[0])
            phases[k.split('.')[0]].append(v[1])
            pers[k.split('.')[0]].append(v[2])
        else:
            amps[k.split('.')[0]] = [v[0]]
            phases[k.split('.')[0]] = [v[1]]
            pers[k.split('.')[0]] = [v[2]]
    return amps, pers, phases


def gen_phase_plot(l,p,n):
     ax = plt.subplot(111, projection='polar')
     for i in l[1].keys():
         ax.scatter(l[2][i], l[1][i],label=i)
     ax.grid(True)
     ax.set_theta_direction(-1)
     ax.set_theta_offset(np.pi/2)
     ax.set_thetagrids(angles=np.linspace(0,360,12), labels=range(0,22,2))
     ax.set_title(n, va='bottom')
     ax.set_rmax(1 + np.floor(max([max(i) for i in list(l[1].values())])))
     ax.set_rmin(np.ceil(min([min(i) for i in list(l[1].values())])-1))
     plt.figtext(0.78, 0.2,'p value of equal \nphases = '+str(round(p,2)))
     plt.legend(bbox_to_anchor=[1.1, 1.1])
     plt.savefig(n+'_phase_v_period.pdf')
     plt.close()

def run_analysis(fname):
         data = pd.read_csv(fname,index_col=0)
         bname = fname[:-4]
         gen_tsplot(data,bname)
         detrended = detrend(data)
         gen_tsplot(detrended,bname+'_detrended')
         stats = get_stats(detrended)
         circdata = [circular(robjects.FloatVector([(j*2*np.pi%22)/22 for j in i]),units="radians", zero=0, rotation='clock') for i in stats[2].values()]
         p = inflect.engine()
         testdata_form =[]
         for i in range(len(circdata)):
             testdata_form.append(p.number_to_words(i)+' = circdata['+str(i)+']')
         testdata_form = ', '.join(testdata_form)
         testdata_form = 'rlist(' + testdata_form + ')'
         testdata_form = testdata_form.replace("-", "")
         testdata = eval(testdata_form)
         wwt_out = wwt(testdata)
         pval = float(wwt_out[3].r_repr())
         gen_phase_plot(stats,pval,bname)

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"h:i:",["help","ifile="])
    except getopt.GetoptError:
        print('residuals.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h',"--help"):
            print('residuals.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
    run_analysis(inputfile)

if __name__ == '__main__':
    main(sys.argv[1:])
