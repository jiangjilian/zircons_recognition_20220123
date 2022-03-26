#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math
from matplotlib.ticker import MaxNLocator
from matplotlib import colors
import matplotlib.ticker as plticker
from globalVar import *

# loc = plticker.MultipleLocator(base=200.0) # this locator puts ticks at regular intervals


# ## ML_JH Zircon plot

# In[3]:


df_ml_jh_plot = pd.read_excel(dataPath + "五张图 Final.xlsx", sheet_name="Sheet2")

# In[4]:


set(df_ml_jh_plot['Type'])

# In[10]:


ml_ziron_type = [
    'I-type detrital zircon', 'I-type Jack Hills zircon', 'S-type detrital zircon', 'S-type Jack Hills zircon'
]

colors_JH = [
    '#FF9587',
    '#FB696A',
    '#9CC1E0',
    '#2F92D7',
]

# In[42]:


bins = range(2500, 4550, 50)

x1 = list(df_ml_jh_plot[df_ml_jh_plot['Type'] == ml_ziron_type[0]]['Age'])
x2 = list(df_ml_jh_plot[df_ml_jh_plot['Type'] == ml_ziron_type[1]]['Age'])
x3 = list(df_ml_jh_plot[df_ml_jh_plot['Type'] == ml_ziron_type[2]]['Age'])
x4 = list(df_ml_jh_plot[df_ml_jh_plot['Type'] == ml_ziron_type[3]]['Age'])
f, ax = plt.subplots(figsize=(6, 3.5))
plt.hist(
    [x1, x2, x3, x4],
    bins=bins,
    stacked=True,
    #     normed=True,
    color=colors_JH,
    #     alpha=0.5,
    edgecolor="w",
    linewidth=0.3,
    label=ml_ziron_type
)

plt.legend(loc='upper right')
plt.xlabel('Age(Ma)', fontsize=13)
plt.ylabel('Frequency', fontsize=13)
# ax.xaxis.set_major_locator(loc)
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=50))
plt.xlim(2500, 4500)
plt.xticks(list(range(2500, 4600, 200)), [str(round(x, 1)) for x in np.arange(2.5, 4.7, 0.2)])
plt.ylim(0, 250)
# plt.xticks(rotation=30)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
# plt.tight_layout()
plt.savefig(figPath + 'sed_stacked_hist0706.pdf')
plt.show()

# ## each location plot

# In[55]:


df_ml_jh_plot = pd.read_excel(dataPath + "五张图 Final.xlsx", sheet_name="Sheet1")

# In[56]:


loc_list = list(set(df_ml_jh_plot['location']))

# In[57]:


loc_list

# In[68]:


bins = range(2500, 4550, 50)
base_list = [50, 5, 2, 2, 5, 5, 1, 5, 2, 2, 50, 10]
fig = plt.figure(figsize=(10, 20))
fig.subplots_adjust(
    hspace=1.2,
    #     wspace=0.2
)
for i, location in zip(range(len(loc_list)), loc_list):
    ax = fig.add_subplot(12, 1, i + 1)
    df_loc = df_ml_jh_plot[df_ml_jh_plot['location'] == location]
    x1 = list(df_loc[df_loc['type'] == 'S-type']['Age'])
    x2 = list(df_loc[df_loc['type'] == 'I-type']['Age'])
    #     f, ax = plt.subplots(figsize=(10,1))
    plt.hist(
        [x1, x2],
        bins=bins,
        stacked=True,
        #     normed=True,
        color=['#FB696A', '#2F92D7'],
        #     alpha=0.5,
        edgecolor="w",
        linewidth=0.3,
        #         label=[]
    )
    #     plt.legend(loc='upper right')
    #     plt.xlabel('Age(Ma)',fontsize=10)
    #     plt.ylabel('Frequency', fontsize=10)
    plt.title(location)
    # ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=base_list[i]))
    plt.xlim(2500, 4500)
    plt.xticks(list(range(2500, 4600, 200)), [str(round(x, 1)) for x in np.arange(2.5, 4.7, 0.2)])
    #     plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    #     plt.xticks(rotation=30)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # plt.tight_layout()
    # plt.savefig('ML_sed_stacked_hist20210506.eps')
#     plt.show()
plt.savefig(figPath + "ML_sed_stacked_hist_50bin_20210706.pdf")

# ## JH_P(REE+Y) vs Hf_plot

# In[2]:


df_JH_plot = pd.read_excel(dataPath + "五张图 Final.xlsx", sheet_name="Sheet4")

type_list = [
    'I-type zircon',
    'I-type detrital zircon',
    'S-type zircon',
    'S-type detrital zircon',
    'I-type TTG zircon'
]
colors = [
    'red',
    'none',
    'steelblue',
    'none',
    '#FFAB44'
]
facecolors = [
    'none',
    'red',
    'none',
    'steelblue',
    'none'
]

# (REE+Y)3 vs Y_plot
# In[5]:


f, ax = plt.subplots(figsize=(4, 3))
for t, c1, c2, tr in zip(type_list, colors, facecolors, [1, .3, 1, .3, 1]):
    sc = plt.scatter(
        df_JH_plot[df_JH_plot["Zircon"] == t]["P (mol%)"],
        df_JH_plot[df_JH_plot["Zircon"] == t]["(REE+Y)3+ (mol%)"],
        s=df_JH_plot[df_JH_plot["Zircon"] == t]["P (μmol/g)"],
        facecolors=c2,
        edgecolors=c1,
        linewidth=0.5,
        alpha=tr,
        label=t
    )
b1 = plt.scatter([], [], s=15, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
b2 = plt.scatter([], [], s=30, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
b3 = plt.scatter([], [], s=45, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
b4 = plt.scatter([], [], s=60, marker='o', facecolors='none', edgecolors='steelblue', linewidth=0.5)
plt.legend((b1, b2, b3, b4),
           ('15 μmol/g', '30 μmol/g', '45 μmol/g', '60 μmol/g'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=8)
plt.xlim(0, 40)
plt.ylim(0, 50)
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.xlabel('P (mol%)', fontsize=13)
plt.ylabel('(REE+Y)3+ (mol%)', fontsize=13)
plt.tight_layout()
plt.savefig(figPath + 'REE_P20210708.pdf')
plt.show()

# ## Age vs P

# In[70]:


df_age_plot = pd.read_excel(dataPath + "五张图 Final.xlsx", sheet_name="Sheet3")

# In[73]:


list(set(df_age_plot['Zircon']))

# In[104]:


ziron_type = [
    'S-type zircon',
    'I-type zircon',
    'S-type zircon (this study)',
    'I-type zircon (this study)',
    'detrital zircon',

]

colors = [
    'steelblue',
    'red',
    'steelblue',
    'red',
    'none'
]

shapes = [
    'o',
    'o',
    'd',
    'd',
    'o'
]

facec = [
    'none',
    'none',
    'none',
    'none',
    'gray'
]

alpha_list = [1, 1, 1, 1, 0.4]

order_list = [10, 10, 10, 10, 1]

f, ax = plt.subplots(figsize=(4, 3))
for t, c, f, s, z, a in zip(ziron_type, colors, facec, shapes, order_list, alpha_list):
    plt.scatter(
        df_age_plot[df_age_plot["Zircon"] == t]["Age（Ma)"] / 1000,
        df_age_plot[df_age_plot["Zircon"] == t]["P (μmol/g)"],
        s=10,
        marker=s,
        facecolors=f,
        edgecolors=c,
        linewidth=0.5,
        alpha=a,
        label=t,
        clip_on=False,
        zorder=z,
    )
lgnd = plt.legend()
for handle in lgnd.legendHandles:
    handle.set_sizes([10.0])
plt.xlim(0, 4.5)
plt.ylim(0, 70)
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=10))
plt.xlabel('Age (Ga)', fontsize=13)
plt.ylabel('P (μmol/g)', fontsize=13)
plt.tight_layout()
plt.savefig(figPath + 'Age_P20210706.pdf')
plt.show()

# ## TSVM plot

# In[115]:


df_tsvm = pd.read_excel(dataPath + "五张图 Final.xlsx", sheet_name="Sheet5")

ziron_type = [
    'detrital zircon', 'S-type zircon', 'I-type zircon'
]

cs = [
    'none',
    'steelblue',
    'red',

]
facecolors = [
    '#FEB76A',
    'none',
    'none',

]
tra = [.5, 1, 1]

f, ax = plt.subplots(figsize=(4, 3))
for t, c1, f, tr in zip(ziron_type, cs, facecolors, tra):
    plt.scatter(
        df_tsvm[df_tsvm["Zircon"] == t]["P (μmol/g)"],
        df_tsvm[df_tsvm["Zircon"] == t]["TSVM score"],
        s=20,
        facecolors=f,
        #         edgecolors=colors.to_rgba(c1, tr),
        edgecolors=c1,
        linewidth=0.5,
        alpha=tr,
        label=t
    )
plt.vlines(15, -15, 15, linestyles="dashed", colors="k", lw=1)
plt.hlines(0, 0, 70, linestyles="dashed", colors="k", lw=1)

lgnd = plt.legend(fontsize=8)
for handle in lgnd.legendHandles:
    handle.set_sizes([20])

plt.xlim(0, 70)
plt.ylim(-15, 15)
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=5))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.xlabel('P (μmol/g)', fontsize=13)
plt.ylabel('TSVM score', fontsize=13)
plt.tight_layout()
plt.savefig(figPath + 'TSVM_Score0706.pdf')
plt.show()


### Plot the ratio of S-type zircons in JH zircons using bootstrap
df_JH = pd.DataFrame(dataPath + "JH paper.xlsx", sheet_name="Bootstrap")

f, ax = plt.subplots(figsize=(4, 3))

#plt.xlim(0, 70)
#plt.ylim(-15, 15)
ax.xaxis.set_major_locator(plticker.MultipleLocator(base=10))
ax.yaxis.set_major_locator(plticker.MultipleLocator(base=5))
ax.tick_params(axis="y", direction="in")
ax.tick_params(axis="x", direction="in")
plt.savefig(figPath + "Ratio of S-type zircons.pdf")
plt.show()