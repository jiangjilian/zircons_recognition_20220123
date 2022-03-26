# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:55:17 2020

@author: 风铃草
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 17:06:40 2020

@author: 风铃草
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as tkr
# import sklearn.bootstrap as bootstrp
from sklearn.utils import resample

import scipy as sp
import os
from scipy.stats import gaussian_kde
import math
from scipy import interpolate
import matplotlib.tri as tri
import seaborn as sns

start_time = 0
end_time = 4500


def compute_time_seq(T, x_col, y_col, Type, result=pd.DataFrame(data=None)):
    global AGE, SiO2, MgO, sampleN, X1, X2, step, X_limited

    AGE = T[x_col]
    Element_data = T[y_col]

    sampleN = int(len(AGE))

    X1 = 4400
    X2 = 4500
    step = 100
    X_limited = 3000

    result = compute_TYPE_mean_std(Element_data, y_col, Type, result)

    return result


def scale_S_ratio(samples):
    count = 0.0
    total = samples.size
    for i in samples:
        if (i == 1):
            count += 1.0
    return count / (total)


def compute_TYPE_mean_std(data, y_col, Type, result):
    low = X1
    high = X2

    nA = [np.nan] * int((X1-X_limited) / step + 2)
    S_num = []

    for j in np.arange(0, int((X1-X_limited) / step + 2), 1):
        # dataAA=[]
        BinAA = data.copy()
        BinAA[BinAA[(AGE < low) | (AGE > high)].index] = np.nan
        dataAA = BinAA[BinAA[~np.isnan(BinAA)].index]
        nA[j] = len(dataAA)
        S_num.append(sum(dataAA))
        # print(nA[j])
        result.loc[j, "AGE_MEDIAN"] = (low + high) / 2  # age

        if nA[j] >= 4:  # less than 4 samples will not be calculated.
            iter = 10000
            S_ratio_list = []
            for i in range(iter):
                bootstrapSamples = resample(dataAA, n_samples=100, replace=1)
                temple_S_ratio = scale_S_ratio(bootstrapSamples)  # single S_ratio
                S_ratio_list.append(temple_S_ratio)

            # CIs = bootstrp.ci(data=dataAA, statfunction=sp.mean, n_samples=10000)
            result.loc[j, str(Type) + " mean"] = np.mean(S_ratio_list)
            result.loc[j, str(Type) + " std"] = np.std(S_ratio_list)  # standard error

        else:
            result.loc[j, str(Type) + " mean"] = np.nan
            result.loc[j, str(Type) + " std"] = np.nan  # standard error

        result.loc[j, "total num"] = nA[j]
        result.loc[j, str(Type) + " num"] = S_num[j]
        low = low - step  # define the bin size (step width)
        high = high - step  # define the bin size (step width)

    return result


def plot_color_line(data, col, Type, color):
    f, ax = plt.subplots(figsize=(20, 10))
    plt.xlim(3.0, 4.5)
    x = data["AGE_MEDIAN"] / 1000
    y = data[str(Type) + " mean"]

    plt.errorbar(x, y, yerr=data[str(Type) + " std"], fmt='-', ecolor='k',
                 elinewidth=2, capsize=2, capthick=1, barsabove=True)

    ax.set_xlabel("AGE(Ga)", fontsize=40, labelpad=10, weight='normal')
    ax.set_ylabel(str(Type), fontsize=40, labelpad=10, weight='normal')
    ax.tick_params(labelsize=35, color='black')
    # ax2.tick_params(labelsize=35, color = 'black',direction='in', length=10, width = 2, colors='black', grid_color='b', pad=10)

    plt.xticks(fontsize=35, color='black')
    plt.yticks(fontsize=35, color='black')
    # plt.legend(loc="upper right",  fontsize=25)
    plt.tight_layout()

    path = figPath + str(Type) + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    plt.savefig(path + "Ratio of S-type zircons.png")


def plot_color_line2(data, col, Type, color):
    fig = plt.figure(figsize=(24, 12))
    ax = fig.add_subplot(111)
    ax.spines['left'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.tick_params(direction='in', length=15, width=2, colors='black', grid_color='b', pad=10, which="major")
    ax.tick_params(direction='in', length=10, width=2, colors='black', grid_color='b', pad=10, which="minor")

    ax.xaxis.set_major_locator(tkr.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(tkr.MultipleLocator(0.25))
    plt.xlim(3.0, 4.5)

    x = data["AGE_MEDIAN"] / 1000
    y = data[str(Type) + " mean"]
    # data[str(index) +" DIFF std"] = data["ICB "+str(index) +" std"] - data["IAB " +str(index) +" std"]
    y_min = y - data[str(Type) + " std"]
    y_max = y + data[str(Type) + " std"]

    bar_width = 0.25
    ax.plot(x, y, color='grey')
    line1 = plt.plot(x, y_min, linestyle="--", color='grey')
    line2 = plt.plot(x, y_max, linestyle="--", color='grey')

    X = [[0, 1], [0, 1]]
    xmin = min(x)
    xmax = max(x)
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]

    ax.fill_between(x, y_min, y_max, color=color, alpha=0.3, label=Type)
    ax.fill_between(x, ymin, y_min, color="white", alpha=1)
    ax.fill_between(x, y_max, ymax, color="white", alpha=1)
    ax.set_aspect('auto')
    # color="red",

    ax.set_xlabel("AGE(Ga)", fontsize=40, labelpad=10, weight='normal')
    ax.set_ylabel(str(Type), fontsize=40, labelpad=10, weight='normal')
    ax.tick_params(labelsize=35, color='black')

    plt.xticks(fontsize=35, color='black')
    plt.yticks(fontsize=35, color='black')
    # plt.legend(loc="upper right",  fontsize=25)
    plt.tight_layout()

    path = figPath + str(Type) + "\\"
    if not os.path.exists(path):
        os.makedirs(path)
    plt.tight_layout()
    fig.savefig(path + "Ratio of S-type zircons2.png")


if __name__ == "__main__":
    dataPath = ".\\data\\"
    figPath = ".\\fig\\"
    bin_width = 100
    fileName = ""
    pred_data = pd.read_excel(dataPath + "JH paper.xlsx", sheet_name="Bootstrap")
    col1 = "Age"
    col2 = "Type"
    S_ratio_seq = compute_time_seq(pred_data, x_col=col1, y_col=col2, Type="S ratio")
    print(S_ratio_seq.columns)
    S_ratio_seq.drop(S_ratio_seq[S_ratio_seq["S ratio" + " mean"] == 0].index)

    plot_color_line(S_ratio_seq, col=col2, Type="S ratio", color="red")
    plot_color_line2(S_ratio_seq, col=col2, Type="S ratio", color="red")
    S_ratio_seq.to_csv(dataPath + "S_ratio_seq.csv")

    plt.show()
