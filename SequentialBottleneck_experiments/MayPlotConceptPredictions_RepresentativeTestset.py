import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

#Load in the raw concept df predictions on the representative test set (for comparison with TCAV scores):
#conceptPredictions_test = pd.read_csv('./SequentialModelOutput/MayRawDensenet121_conceptPredictions_RepresentativeTestset.csv',index_col = 'Unnamed: 0')
#Inspect the FGADR test set instead:
conceptPredictions_test = pd.read_csv('./SequentialModelOutput/MayRawDensenet121_conceptPredictions_FGADRTestset.csv',index_col = 'Unnamed: 0')

#Counting number of positive predictions for MA
MA_level0_predictions = 0
MA_level1_predictions = 0
MA_level2_predictions= 0
MA_level3_predictions = 0
MA_level4_predictions = 0
#Repeat for HE:
HE_level0_predictions = 0
HE_level1_predictions = 0
HE_level2_predictions= 0
HE_level3_predictions = 0
HE_level4_predictions = 0
#Soft exudates:
SoftEx_level0_predictions = 0
SoftEx_level1_predictions = 0
SoftEx_level2_predictions= 0
SoftEx_level3_predictions = 0
SoftEx_level4_predictions = 0
#Hard exudates:
HardEx_level0_predictions = 0
HardEx_level1_predictions = 0
HardEx_level2_predictions= 0
HardEx_level3_predictions = 0
HardEx_level4_predictions = 0
#NV:
NV_level0_predictions = 0
NV_level1_predictions = 0
NV_level2_predictions= 0
NV_level3_predictions = 0
NV_level4_predictions = 0
#IRMA:
IRMA_level0_predictions = 0
IRMA_level1_predictions = 0
IRMA_level2_predictions= 0
IRMA_level3_predictions = 0
IRMA_level4_predictions = 0

#Count the number of observations for each DR level: 
level0_count = 0
level1_count = 0
level2_count = 0
level3_count = 0
level4_count = 0
for i in range(conceptPredictions_test.shape[0]):
    #Get the DR level
    dr_level = conceptPredictions_test.iloc[i,-1]
    print('Level of DR:',dr_level)
    #Get the raw concept predictions:
    concept_data = conceptPredictions_test.iloc[i,1]
    #Since these are (of unknown causes) interpreted as a string-list
    #We need to convert them to a proper list of float-values:
    concept_data = concept_data.strip('"')
    concept_data = concept_data.strip('[]')
    concept_data = list(concept_data.split(','))
    concept_data = list(map(float,concept_data))
    print('Concept data:',concept_data)
    ma_concept = concept_data[0]
    he_concept = concept_data[1]
    softEx_concept = concept_data[2]
    hardEx_concept = concept_data[3]
    nv_concept = concept_data[4]
    irma_concept = concept_data[5]
    print('MA concept:',ma_concept)
    print('NV concept:',nv_concept)
    if dr_level == 0:
        level0_count+=1
        if ma_concept>=0:
            MA_level0_predictions+=1
        if he_concept>=0:
            HE_level0_predictions+=1
        if softEx_concept>=0:
            SoftEx_level0_predictions+=1
        if hardEx_concept>=0:
            HardEx_level0_predictions+=1
        if nv_concept>=0:
            NV_level0_predictions+=1
        if irma_concept>=0:
            IRMA_level0_predictions+=1
    elif dr_level ==1:
        level1_count+=1
        if ma_concept>=0:
            MA_level1_predictions+=1
        if he_concept>=0:
            HE_level1_predictions+=1
        if softEx_concept>=0:
            SoftEx_level1_predictions+=1
        if hardEx_concept>=0:
            HardEx_level1_predictions+=1
        if nv_concept>=0:
            NV_level1_predictions+=1
        if irma_concept>=0:
            IRMA_level1_predictions+=1
    elif dr_level == 2:
        level2_count+=1
        if ma_concept>=0:
            MA_level2_predictions+=1
        if he_concept>=0:
            HE_level2_predictions+=1
        if softEx_concept>=0:
            SoftEx_level2_predictions+=1
        if hardEx_concept>=0:
            HardEx_level2_predictions+=1
        if nv_concept>=0:
            NV_level2_predictions+=1
        if irma_concept>=0:
            IRMA_level2_predictions+=1
    elif dr_level == 3:
        level3_count+=1
        if ma_concept>=0:
            MA_level3_predictions+=1
        if he_concept>=0:
            HE_level3_predictions+=1
        if softEx_concept>=0:
            SoftEx_level3_predictions+=1
        if hardEx_concept>=0:
            HardEx_level3_predictions+=1
        if nv_concept>=0:
            NV_level3_predictions+=1
        if irma_concept>=0:
            IRMA_level3_predictions+=1
    elif dr_level == 4:
        level4_count+=1
        if ma_concept>=0:
            MA_level4_predictions+=1
        if he_concept>=0:
            HE_level4_predictions+=1
        if softEx_concept>=0:
            SoftEx_level4_predictions+=1
        if hardEx_concept>=0:
            HardEx_level4_predictions+=1
        if nv_concept>=0:
            NV_level4_predictions+=1
        if irma_concept>=0:
            IRMA_level4_predictions+=1
    
print('MA for DR level 0:', MA_level0_predictions)
print('HE for DR level 0:',HE_level0_predictions)
print('SoftEx for DR level 0:',SoftEx_level0_predictions)
print('HardEx for DR level 0:',HardEx_level0_predictions)
print('NV for DR level 0:',NV_level0_predictions)
print('IRMA for DR level 0:',IRMA_level0_predictions)

print('MA for DR level 1:', MA_level1_predictions)
print('HE for DR level 1:',HE_level1_predictions)
print('SoftEx for DR level 1:',SoftEx_level1_predictions)
print('HardEx for DR level 1:',HardEx_level1_predictions)
print('NV for DR level 1:',NV_level1_predictions)
print('IRMA for DR level 1:',IRMA_level1_predictions)

print('MA for DR level 2:', MA_level2_predictions)
print('HE for DR level 2:',HE_level2_predictions)
print('SoftEx for DR level 2:',SoftEx_level2_predictions)
print('HardEx for DR level 2:',HardEx_level2_predictions)
print('NV for DR level 2:',NV_level2_predictions)
print('IRMA for DR level 2:',IRMA_level2_predictions)

print('MA for DR level 3:', MA_level3_predictions)
print('HE for DR level 3:',HE_level3_predictions)
print('SoftEx for DR level 3:',SoftEx_level3_predictions)
print('HardEx for DR level 3:',HardEx_level3_predictions)
print('NV for DR level 3:',NV_level3_predictions)
print('IRMA for DR level 3:',IRMA_level3_predictions)

print('MA for DR level 4:', MA_level4_predictions)
print('HE for DR level 4:',HE_level4_predictions)
print('SoftEx for DR level 4:',SoftEx_level4_predictions)
print('HardEx for DR level 4:',HardEx_level4_predictions)
print('NV for DR level 4:',NV_level4_predictions)
print('IRMA for DR level 4:',IRMA_level4_predictions)

print('Total number of level 0 images:',level0_count)
print('Total number of level 1 images:',level1_count)
print('Total number of level 2 images:',level2_count)
print('Total number of level 3 images:',level3_count)
print('Total number of level 4 images:',level4_count)

#plot barcharts for each DR level
num_concepts = 6
bar_width = 0.35
# create location for each bar. scale by an appropriate factor to ensure 
# the final plot doesn't have any parts overlapping
index = np.arange(num_concepts) * bar_width
print('Index:',index)
my_colors = mcolors.TABLEAU_COLORS
names = list(my_colors)
print('Colors to choose:',names)
bar_x = []
plot_conceptNames = ['MA','HE','EX','SE','IRMA','NV']
#Divide by number of images (50 for each DR level) to get the percentage concept count ranging from 0 to 1
level_0_conceptCounts = [MA_level0_predictions/level0_count,HE_level0_predictions/level0_count,HardEx_level0_predictions/level0_count,SoftEx_level0_predictions/level0_count,IRMA_level0_predictions/level0_count,NV_level0_predictions/level0_count]
level_1_conceptCounts = [MA_level1_predictions/level1_count,HE_level1_predictions/level1_count,HardEx_level1_predictions/level1_count,SoftEx_level1_predictions/level1_count,IRMA_level1_predictions/level1_count,NV_level1_predictions/level1_count]
level_2_conceptCounts = [MA_level2_predictions/level2_count,HE_level2_predictions/level2_count,HardEx_level2_predictions/level2_count,SoftEx_level2_predictions/level2_count,IRMA_level2_predictions/level2_count,NV_level2_predictions/level2_count]
level_3_conceptCounts = [MA_level3_predictions/level3_count,HE_level3_predictions/level3_count,HardEx_level3_predictions/level3_count,SoftEx_level3_predictions/level3_count,IRMA_level3_predictions/level3_count,NV_level3_predictions/level3_count]
level_4_conceptCounts = [MA_level4_predictions/level4_count,HE_level4_predictions/level4_count,HardEx_level4_predictions/level4_count,SoftEx_level4_predictions/level4_count,IRMA_level4_predictions/level4_count,NV_level4_predictions/level4_count]
for i in range(6):
    bar_x.append(i*bar_width)
#Plotting the predicted concept counts for DR levels 1-4 
#In order to compare with the TCAV scores
fig, ax = plt.subplots(1,4, figsize=(32,8))

#https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/subplots_demo.html#sphx-glr-gallery-subplots-axes-and-figures-subplots-demo-py
#DR level 1
ax[0].bar(bar_x, level_1_conceptCounts, bar_width, label=plot_conceptNames, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[0].set_title('DR level 1',fontsize=32)
#ax[0,0].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[0].set_xticks(bar_x)
ax[0].set_xticklabels(plot_conceptNames, rotation = 75,fontsize=32)
ax[0].set_ylim((0,1.05))

#DR level 2
ax[1].bar(bar_x, level_2_conceptCounts, bar_width, label=plot_conceptNames, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[1].set_title('DR level 2',fontsize=32)
#ax[0,0].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[1].set_xticks(bar_x)
ax[1].set_xticklabels(plot_conceptNames, rotation = 75,fontsize=32)
ax[1].set_ylim((0,1.05))

#DR level 3
ax[2].bar(bar_x, level_3_conceptCounts, bar_width, label=plot_conceptNames, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[2].set_title('DR level 3',fontsize=32)
#ax[0,0].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[2].set_xticks(bar_x)
ax[2].set_xticklabels(plot_conceptNames, rotation = 75,fontsize=32)
ax[2].set_ylim((0,1.05))

#DR level 4
ax[3].bar(bar_x, level_4_conceptCounts, bar_width, label=plot_conceptNames, 
    color=['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown'])
    #color=['blue','orange','green','red','brown','purple'])
ax[3].set_title('DR level 4',fontsize=32)
#ax[0,0].set_ylabel('TCAV Score')
#ax.set_xticks(index * bar_width / 2)
ax[3].set_xticks(bar_x)
ax[3].set_xticklabels(plot_conceptNames, rotation = 75,fontsize=32)
ax[3].set_ylim((0,1.05))

ax[0].set_ylabel('Concept count',fontsize=32)
ax[0].set_yticklabels([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=32)
# Hide x labels and tick labels for top plots and y ticks for right plots.
ax[1].label_outer()
ax[2].label_outer()
ax[3].label_outer()

#fig.tight_layout()
#plt.show()
#Shrink the space between the subplots:
plt.subplots_adjust(wspace=0.1)
plt.savefig('PlotBottleneckConceptCounts_FGADRTestset_subplotsMayLong.png', bbox_inches = 'tight')
