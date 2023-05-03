import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
#Create a dict with the concept and a list of 
#TCAV mean + std, and whether the concept was significant or not
#If not significant, the values are just set to 0
'''
#On the full images:
results_class0 = {
    'MA': [0.39, 0.319, True],
    'HE': [0.674, 0.306, True],
    'EX': [0, 0, False],
    'SE': [0.208, 0.25, True],
    'IRMA':[0.354, 0.264, True],
    'NV': [0.145, 0.13, True]
}

results_class1 = {
    'MA': [0.652, 0.325, True],
    'HE': [0.242, 0.161, True],
    'EX': [0.232, 0.221, True],
    'SE': [0, 0, False],
    'IRMA':[0, 0, False],
    'NV': [0.256, 0.224, True]
}
results_class2 = {
    'MA': [0.725, 0.277, True],
    'HE': [0, 0, False],
    'EX': [0.346, 0.339, True],
    'SE': [0.87, 0.18, True],
    'IRMA':[0.898, 0.183, True],
    'NV': [0.134, 0.208, True]
}

results_class3 = {
    'MA': [0.615, 0.307, True],
    'HE': [0.907, 0.164, True],
    'EX': [0.745, 0.281, True],
    'SE': [0.64, 0.282, True],
    'IRMA':[0.728, 0.228, True],
    'NV': [0.073, 0.103, True]
}

results_class4 = {
    'MA': [0, 0, False],
    'HE': [0.756, 0.208, True],
    'EX': [0.877, 0.211, True],
    'SE': [0, 0, False],
    'IRMA':[0, 0, False],
    'NV': [0.999, 0.004, True]
}
'''
#On masked images:
results_class0 = {
    'MA': [0, 0, False],
    'HE': [0, 0, False],
    'EX': [0.401, 0.283, True],
    'SE': [0.38, 0.277, True],
    'IRMA':[0.7, 0.278, True],
    'NV': [0.2, 0.226, True]
}

results_class1 = {
    'MA': [0.602, 0.239, True],
    'HE': [0.598, 0.264, True],
    'EX': [0.263, 0.197, True],
    'SE': [0, 0, False],
    'IRMA':[0.697, 0.198, True],
    'NV': [0.255, 0.234, True]
}
results_class2 = {
    'MA': [0.722, 0.291, True],
    'HE': [0, 0, False],
    'EX': [0.292, 0.309, True],
    'SE': [0.845, 0.258, True],
    'IRMA':[0.912, 0.198, True],
    'NV': [0.113, 0.142, True]
}

results_class3 = {
    'MA': [0, 0, False],
    'HE': [0.887, 0.095, True],
    'EX': [0.923, 0.085, True],
    'SE': [0.894, 0.163, True],
    'IRMA':[0.715, 0.273, True],
    'NV': [0.016, 0.033, True]
}

results_class4 = {
    'MA': [0.361, 0.322, True],
    'HE': [0.777, 0.259, True],
    'EX': [0.79, 0.244, True],
    'SE': [0, 0, False],
    'IRMA':[0.36, 0.277, True],
    'NV': [1.0, 0.0, True]
}

num_concepts = 6
bar_width = 0.35
# create location for each bar. scale by an appropriate factor to ensure 
# the final plot doesn't have any parts overlapping
index = np.arange(num_concepts) * bar_width
print('Index:',index)
my_colors = mcolors.TABLEAU_COLORS
names = list(my_colors)
print('Colors to choose:',names)

fig, ax = plt.subplots(1,4, figsize=(32,8))
#Values for DR level 1
plot_concepts1 = []
TCAV_means1 = []
TCAV_std1 = []
TCAV_significance1 = []
bar_x1 = []
for i, [concept_name, vals] in enumerate(results_class1.items()):
    #The TCAV mean is vals[0], the TCAV std is vals[1]
    TCAV_means1.append(vals[0])
    TCAV_std1.append(vals[1])
    bar_x1.append(i * bar_width)
    plot_concepts1.append(concept_name)
    TCAV_significance1.append(vals[2])
text_sequence1=[]
for j in TCAV_significance1:
    if not j:
        text_sequence1.append('*')
    else:
        text_sequence1.append(' ')

#Values for DR level 2
plot_concepts2 = []
TCAV_means2 = []
TCAV_std2 = []
TCAV_significance2 = []
bar_x2 = []
for i, [concept_name, vals] in enumerate(results_class2.items()):
    #The TCAV mean is vals[0], the TCAV std is vals[1]
    TCAV_means2.append(vals[0])
    TCAV_std2.append(vals[1])
    bar_x2.append(i * bar_width)
    plot_concepts2.append(concept_name)
    TCAV_significance2.append(vals[2])
text_sequence2=[]
for j in TCAV_significance2:
    if not j:
        text_sequence2.append('*')
    else:
        text_sequence2.append(' ')

#Values for DR level 3
plot_concepts3 = []
TCAV_means3 = []
TCAV_std3 = []
TCAV_significance3 = []
bar_x3 = []
for i, [concept_name, vals] in enumerate(results_class3.items()):
    #The TCAV mean is vals[0], the TCAV std is vals[1]
    TCAV_means3.append(vals[0])
    TCAV_std3.append(vals[1])
    bar_x3.append(i * bar_width)
    plot_concepts3.append(concept_name)
    TCAV_significance3.append(vals[2])
text_sequence3=[]
for j in TCAV_significance3:
    if not j:
        text_sequence3.append('*')
    else:
        text_sequence3.append(' ')

#Values for DR level 4
plot_concepts4 = []
TCAV_means4 = []
TCAV_std4 = []
TCAV_significance4 = []
bar_x4 = []
for i, [concept_name, vals] in enumerate(results_class4.items()):
    #The TCAV mean is vals[0], the TCAV std is vals[1]
    TCAV_means4.append(vals[0])
    TCAV_std4.append(vals[1])
    bar_x4.append(i * bar_width)
    plot_concepts4.append(concept_name)
    TCAV_significance4.append(vals[2])
text_sequence4=[]
for j in TCAV_significance4:
    if not j:
        text_sequence4.append('*')
    else:
        text_sequence4.append(' ')

#Now, to the subplots themselves:
#https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-demo-py
ax[0].bar(bar_x1, TCAV_means1, bar_width, yerr=TCAV_std1, label=plot_concepts1, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[0].set_title('DR level 1',fontsize=32)
#ax[0,0].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[0].set_xticks(bar_x1)
ax[0].set_xticklabels(plot_concepts1, rotation = 75,fontsize=32)
ax[0].set_ylim((0,1.13))
for i in range(6):
    ax[0].text(bar_x1[i]-0.03,0.01,text_sequence1[i],fontdict = {'weight': 'bold', 'size': 32})

#DR level 2:
ax[1].bar(bar_x2, TCAV_means2, bar_width, yerr=TCAV_std2, label=plot_concepts2, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[1].set_title('DR level 2',fontsize=32)
#ax[0,1].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[1].set_xticks(bar_x2)
ax[1].set_xticklabels(plot_concepts1, rotation = 75,fontsize=32)
ax[1].set_ylim((0,1.13))
for i in range(6):
    ax[1].text(bar_x2[i]-0.03,0.01,text_sequence2[i],fontdict = {'weight': 'bold', 'size': 32})

#DR level 3
ax[2].bar(bar_x3, TCAV_means3, bar_width, yerr=TCAV_std3, label=plot_concepts3, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[2].set_title('DR level 3',fontsize=32)
#ax[1,1].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[2].set_xticks(bar_x3)
ax[2].set_xticklabels(plot_concepts3, rotation = 75,fontsize=32)
ax[2].set_ylim((0,1.13))
for i in range(6):
    ax[2].text(bar_x3[i]-0.03,0.01,text_sequence3[i],fontdict = {'weight': 'bold', 'size': 32})

#DR level 4
ax[3].bar(bar_x4, TCAV_means4, bar_width, yerr=TCAV_std4, label=plot_concepts4, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[3].set_title('DR level 4',fontsize=32)
#ax[1,0].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[3].set_xticks(bar_x4)
ax[3].set_xticklabels(plot_concepts4, rotation = 75,fontsize=32)
ax[3].set_ylim((0,1.13))
for i in range(6):
    ax[3].text(bar_x4[i]-0.03,0.01,text_sequence4[i],fontdict = {'weight': 'bold', 'size': 32})


#ax[1,0].set(ylabel='TCAV Score')
#ax[0].set(ylabel='TCAV Score',fontsize=20)
ax[0].set_ylabel('TCAV Score',fontsize=32)
ax[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=32)
# Hide x labels and tick labels for top plots and y ticks for right plots.
ax[1].label_outer()
ax[2].label_outer()
ax[3].label_outer()

#fig.tight_layout()
#plt.show()
#Shrink the space between the subplots:
plt.subplots_adjust(wspace=0.1)
plt.savefig('PlotTCAVscores_45_test20_subplotsMaskedAprilLong.png', bbox_inches = 'tight')
