import os
import random
import argparse
import torch
import copy

import numpy as np
from torch import functional
import torch.nn as nn

from sklearn import metrics
from scipy.stats import ttest_ind

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from captum.concept import TCAV, Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from PIL import Image
import matplotlib.pyplot as plt
import glob
#And import albumentations for CLAHE preprocessing:
import cv2 as cv
import albumentations as albu
from albumentations.pytorch import ToTensorV2
#Must customize the dataset to use albumentations...
from torch.utils.data import Dataset as BaseDataset

#Try some monkey patching for getting the layer code to work:
from captum.concept import CAV
from captum.concept._core import tcav
from captum._utils.av import AV
from typing import Any, List, Optional, Tuple, Union
from torch import Tensor
#from sklearn.linear_model import SGDClassifier

#Device (I use GPU 0, and Vajira use 1):
#torch.cuda.set_device(0)
#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Can try to use both GPUs to avoid error:
#https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
#https://stackoverflow.com/questions/54216920/how-to-use-multiple-gpus-in-pytorch 

#Since the GPUs are not working well here (memory issues), I use cpu:
DEVICE = torch.device("cpu")
n_classes = 5
#Create dataloaders for the positive and the negative examples
#Some help is provided here: https://captum.ai/api/concept.html#classifier
#NB! Check out this tutorial!
# https://captum.ai/tutorials/TCAV_Image 

#Use CLAHE for increased image quality:
def transform_val_clahe(img):
    transform_clahe = albu.Compose([albu.CLAHE(clip_limit=2.0,p=1.0),
        albu.Resize(620,620),
        albu.Normalize(mean = (0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225),p=1.0),
        ToTensorV2()
    ])
    return transform_clahe(image=img)["image"]


def get_tensor_from_filename(filename):
    #img = Image.open(filename).convert("RGB")
    #return transform_val_fund(img)
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return transform_val_clahe(img)

def load_image_tensors(class_name, root_path='ConceptFoldersDiaretDB/BalancedConcepts', transform=True):
    path = os.path.join(root_path, class_name)
    #Since the images have four(!) different formats:
    filenames = glob.glob(path + '/*.png')
    filenames2 = glob.glob(path + '/*.jpg')
    filenames3 = glob.glob(path + '/*.jpeg')
    filenames4 = glob.glob(path + '/*.JPG')
    filenames = filenames + filenames2 + filenames3 + filenames4
    print('Number of images that are loaded:',len(filenames))
    tensors = []
    for filename in filenames:
        #img = Image.open(filename).convert('RGB')
        #tensors.append(transform_val_fund(img) if transform else img)
        img = cv.imread(filename)
        img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        tensors.append(transform_val_clahe(img) if transform else img)
    return tensors

def assemble_concept(name, id, concepts_path="ConceptFoldersDiaretDB/BalancedConcepts"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=name, data_iter=concept_iter)

#For densenet, we must define our own assemble_save_path since the layers are a bit strange
#https://stackoverflow.com/questions/10429547/how-to-change-a-function-in-existing-3rd-party-library-in-python
#https://stackoverflow.com/questions/5626193/what-is-monkey-patching
#The original function is here: 
#https://github.com/pytorch/captum/blob/master/captum/concept/_core/cav.py
#The new function below will work for Denseblock 2, 3 and 4
def assemble_save_path(path, model_id, concepts, layer):
        r"""
        A utility method for assembling filename and its path, from
        a concept list and a layer name.
        Args:
            path (str): A path to be concatenated with the concepts key and
                    layer name.
            model_id (str): A unique model identifier associated with input
                    `layer` and `concepts`
            concepts (list[Concept]): A list of concepts that are concatenated
                    together and used as a concept key using their ids. These
                    concept ids are retrieved from TCAV s`Concept` objects.
            layer (str): The name of the layer for which the activations are
                    computed.
        Returns:
            cav_path(str): A string containing the path where the computed CAVs
                    will be stored.
                    For example, given:
                        concept_ids = [0, 1, 2]
                        concept_names = ["striped", "random_0", "random_1"]
                        layer = "inception4c"
                        path = "/cavs",
                    the resulting save path will be:
                        "/cavs/default_model_id/0-1-2-inception4c.pkl"
        """
        #file_name = concepts_to_str(concepts) + "-" + layer + ".pkl"
        #print('Layername for savepath function:',layer)
        #Replace the layer with the layer of interest for Densenet
        last_layername = list(layer.keys())[-1]
        if last_layername == 'denselayer16':
            block_name = 'denseblock4'
        elif last_layername == 'denselayer24':
            block_name = 'denseblock3'
        elif last_layername == 'denselayer12':
            block_name = 'denseblock2'
        file_name = concepts_to_str(concepts) + "-" + block_name + ".pkl"

        return os.path.join(path, model_id, file_name)
CAV.assemble_save_path = assemble_save_path

#Also for the _get_module_by_name
#Original function here: https://github.com/pytorch/captum/blob/master/captum/_utils/common.py
def _get_module_from_nameNew(model, layer):
    r"""
    Returns the module (layer) object, given its (string) name
    in the model.
    Args:
            name (str): Module or nested modules name string in self.model
    Returns:
            The module (layer) in self.model.
    """
    #return reduce(getattr, layer_name.split("."), model)
    #return model._modules['features']._modules['denseblock4']
    #Try to simplify it even more:
    return layer
tcav._get_module_from_name = _get_module_from_nameNew


#Might need to change the AV.exists code, since it doesn't understand the densenet layers:
#Original code in captum/_utils/av.py, line 146
#https://github.com/pytorch/captum/blob/master/captum/_utils/av.py
@staticmethod
#def existsNew(path,model_id,identifier,layer,num_id):
def existsNew(
        path: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> bool:
        r"""
        Verifies whether the model + layer activations exist
        under the path.
        Args:
            path (str): The path where the activation vectors
            for the `model_id` are stored.
            model_id (str): The name/version of the model for which layer activations
            are being computed and stored.
            identifier (str or None): An optional identifier for the layer activations.
            Can be used to distinguish between activations for different
            training batches. For example, the id could be a suffix composed of
            a train/test label and numerical value, such as "-train-xxxxx".
            The numerical id is often a monotonic sequence taken from datetime.
            layer (str or None): The layer for which the activation vectors are
                computed.
            num_id (str): An optional string representing the batch number for which
                the activation vectors are computed
            Returns:
                exists (bool): Indicating whether the activation vectors for the `layer`
                    and `identifier` (if provided) and num_id (if provided) were stored
                    in the manifold. If no `identifier` is provided, will return `True`
                    if any layer activation exists, whether it has an identifier or
                    not, and vice-versa.
        """
        av_dir = AV._assemble_model_dir(path, model_id)
        layer_dir = os.path.join(av_dir,layer)
        #print('AV dir:',av_dir)
        #print('Layer:',layer)
        #print('Num_id:',num_id)
        last_layername = list(identifier.keys())[-1]
        if last_layername == 'denselayer16':
            my_identifier = 'denseblock4'
        elif last_layername == 'denselayer24':
            my_identifier = 'denseblock3'
        elif last_layername == 'denselayer12':
            my_identifier = 'denseblock2'
        #print('My Identifier:',my_identifier)
        #my_identifier = None
        av_filesearch = AV._construct_file_search(
            path, model_id, my_identifier, layer, num_id
        )
        return os.path.exists(av_dir) and len(glob.glob(av_filesearch)) > 0
        #return os.path.exists(av_dir) and os.path.exists(layer_dir)
AV.exists = existsNew


#Also inspect the AV.save function:
#Line 186 in captum/_utils/av.py
#https://github.com/pytorch/captum/blob/master/captum/_utils/av.py
@staticmethod
def saveNew(
    path,
    model_id,
    identifier,
    layers,
    act_tensors,
    num_id,
    ) -> None:

        r"""
        Saves the activation vectors `act_tensor` for the
        `layer` under the manifold `path`.
        Args:
            path (str): The path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the identifier could be
                    a suffix composed of a train/test label and numerical value, such
                    as "-src-abc".
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in
                    AV.generate_dataset_activations from batch index.
                    It assumes identifier is same for all layers if a list of
                    `layers` is provided.
            layers (str or list[str]): The layer(s) for which the activation vectors
                    are computed.
            act_tensors (tensor or list of tensor): A batch of activation vectors.
                    This must match the dimension of `layers`.
            num_id (str): string representing the batch number for which the activation
                    vectors are computed
        """
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(act_tensors, Tensor):
            act_tensors = [act_tensors]
        #Andrea: Added some extra code here:
        #print('Activation tensors:')
        #print('Length:',len(act_tensors))
        #print(act_tensors)
        #print('Length of layers:',len(layers))
        block_names = []
        last_layername = list(layers._modules.keys())[-1]
        if last_layername == 'denselayer16':
            block_name = 'denseblock4'
        elif last_layername == 'denselayer24':
            block_name = 'denseblock3'
        elif last_layername == 'denselayer12':
            block_name = 'denseblock2'
        block_names.append(block_name)
        #print('Blockname:',block_name)
        #if len(layers) != len(act_tensors):
        if len(block_names) != len(act_tensors):
            raise ValueError("The dimension of `layers` and `act_tensors` must match!")

        av_dir = AV._assemble_model_dir(path, model_id)

        #for i, layer in enumerate(layers):
        for i, layer in enumerate(block_names):
            av_save_fl_path = os.path.join(
                AV._assemble_file_path(av_dir, identifier, layer), "%s.pt" % num_id
            )

            layer_dir = os.path.dirname(av_save_fl_path)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)
            torch.save(act_tensors[i], av_save_fl_path)
AV.save = saveNew

#And the AV.load function
#Line 242 in captum/_utils/av.py
#https://github.com/pytorch/captum/blob/master/captum/_utils/av.py
@staticmethod
def loadNew(
    path: str,
    model_id: str,
    identifier: Optional[str] = None,
    layer: Optional[str] = None,
    num_id: Optional[str] = None,
):
    r"""
    Loads lazily the activation vectors for given `model_id` and
    `layer` saved under the `path`.
    Args:
        path (str): The path where the activation vectors
                for the `layer` are stored.
        model_id (str): The name/version of the model for which layer activations
                are being computed and stored.
        identifier (str or None): An optional identifier for the layer
                activations. Can be used to distinguish between activations for
                different training batches.
        layer (str or None): The layer for which the activation vectors
            are computed.
        num_id (str): An optional string representing the batch number for which
                the activation vectors are computed
    Returns:
        dataset (AV.AVDataset): AV.AVDataset that allows to iterate
                    over the activation vectors for given layer, identifier (if
                    provided), num_id (if provided).  Returning an AV.AVDataset as
                    opposed to a DataLoader constructed from it offers more
                    flexibility.  Raises RuntimeError if activation vectors are not
                    found.
        """

    av_save_dir = AV._assemble_model_dir(path, model_id)
    print('Saving directory:', av_save_dir)
    #print(layer)
    last_layername = list(layer._modules.keys())[-1]
    #print('Last layername for loading TCAV:',last_layername)
    #print('The identifier:',identifier)
    if last_layername == 'denselayer16':
            block_name = 'denseblock4'
    elif last_layername == 'denselayer24':
        block_name = 'denseblock3'
    elif last_layername == 'denselayer12':
        block_name = 'denseblock2'
    #print('For loading the AV, here is corresponding block name:',block_name)
    if os.path.exists(av_save_dir):
        #avdataset = AV.AVDataset(path, model_id, identifier, layer, num_id)
        avdataset = AV.AVDataset(path, model_id, identifier, block_name, num_id)
        return avdataset
    else:
        raise RuntimeError(
            f"Activation vectors for model {model_id} was not found at path {path}"
        )
AV.load = loadNew


#Just give the entire image:
NV_concept = assemble_concept("NV/PositiveExamples", 0, concepts_path="ConceptFoldersAll")
random_0_concept = assemble_concept("NV/NegativeExamples1",1,concepts_path="ConceptFoldersAll")
random_1_concept = assemble_concept("NV/NegativeExamples2",2,concepts_path="ConceptFoldersAll")
random_2_concept = assemble_concept("NV/NegativeExamples3",3,concepts_path="ConceptFoldersAll")
random_3_concept = assemble_concept("NV/NegativeExamples4",4,concepts_path="ConceptFoldersAll")
random_4_concept = assemble_concept("NV/NegativeExamples5",5,concepts_path="ConceptFoldersAll")
MA_concept = assemble_concept("MA/PositiveExamples", 6, concepts_path="ConceptFoldersAll")
HE_concept = assemble_concept("HE/PositiveExamples", 7, concepts_path="ConceptFoldersAll")
HardEx_concept = assemble_concept("HardExudates/PositiveExamples", 8, concepts_path="ConceptFoldersAll")
SoftEx_concept = assemble_concept("SoftExudates/PositiveExamples", 9, concepts_path="ConceptFoldersAll")
IRMA_concept = assemble_concept("irma/PositiveExamples", 10, concepts_path="ConceptFoldersAll")


random_concepts = [random_2_concept,random_3_concept,random_4_concept]
experimental_sets = [[NV_concept, random_0_concept],[NV_concept,random_1_concept]]
experimental_sets.extend([[NV_concept, random_concept] for random_concept in random_concepts])
experimental_sets.append([random_0_concept,random_1_concept])
experimental_sets.extend([[random_0_concept,random_concept] for random_concept in random_concepts])
print('List of experimental concepts:')
print(experimental_sets)

#Now, let's define a convenience function for assembling the experiments together 
#as lists of Concept objects, creating and running the TCAV:
def assemble_scores(scores, experimental_sets, idx, score_layer, score_type):
    score_list = []
    print('Score layer:',score_layer)
    print('Score type:',score_type)
    for concepts in experimental_sets:
        score_list.append(scores["-".join([str(c.id) for c in concepts])][score_layer][score_type][idx])
    print('Score list:',score_list)
    return score_list
#And a function to look at p-values
#We label concept populations as overlapping if p-value > 0.05 otherwise disjoint.
def get_pval(scores, experimental_sets, score_layer, score_type, alpha=0.05, print_ret=False):
    
    P1 = assemble_scores(scores, experimental_sets, 0, score_layer, score_type)
    P2 = assemble_scores(scores, experimental_sets, 1, score_layer, score_type)
    
    if print_ret:
        print('P1[mean, std]: ', format_float(np.mean(P1)), format_float(np.std(P1)))
        print('P2[mean, std]: ', format_float(np.mean(P2)), format_float(np.std(P2)))

    _, pval = ttest_ind(P1, P2)

    if print_ret:
        print("p-values:", format_float(pval))

    if pval < alpha:    # alpha value is 0.05 or 5%
        relation = "Disjoint"
        if print_ret:
            print("Disjoint")
    else:
        relation = "Overlap"
        if print_ret:
            print("Overlap")
        
    return P1, P2, format_float(pval), relation

model = models.densenet121()
model.classifier = nn.Linear(model.classifier.in_features, n_classes)

#model = models.inception_v3()
#model.fc = nn.Linear(model.fc.in_features, n_classes)
chkpoint_path = '../output/CroppedKaggle_Densenet121_100epochs.pt'
#chkpoint_path = '../output/CroppedKaggle_InceptionV3_EarlyStopping.pt'
chkpoint = torch.load(chkpoint_path, map_location = 'cpu')
model.load_state_dict(chkpoint)

model.to(DEVICE)
model.eval()

#For Densenet, we need to access the model features to get to the conv layers
model_features = model._modules['features']
#When running in parallel:
print('Hi!')

#Warning when using the default classifier, since all data must be in memory at the same time
# https://captum.ai/api/_modules/captum/concept/_utils/classifier.html
mytcav = TCAV(model = model, layers = [model_features._modules['denseblock2'],model_features._modules['denseblock3'],model_features._modules['denseblock4']])
#If InceptionV3:
#mytcav = TCAV(model = model, layers = ['Mixed_6d','Mixed_6e','Mixed_7c'])

#Load the cropped test images for DR level 4:
#fgadr_imgs = load_image_tensors('4',root_path='../Data/CroppedData/CroppedTestCombinedXL', transform=False)
#For Densenet121, only compute TCAV scores for the smaller test set due to memory issues...
fgadr_imgs = load_image_tensors('4',root_path='./RepresentativeTestFolderKaggleCropped', transform=False)
fgadr_tensors = torch.stack([transform_val_clahe(img) for img in fgadr_imgs])

experimental_sets = [[MA_concept,HE_concept,HardEx_concept,SoftEx_concept,IRMA_concept,NV_concept]]

scores = mytcav.interpret(inputs=fgadr_tensors.to(DEVICE),
                                        experimental_sets=experimental_sets,
                                        target=4 #The target class (DR level 4)
                                        #n_steps=5, #This is for the LayerIntegratedGradients: steps used by the approx. method (default is 50)
                                       )
#tcav_scores_w_random = mytcav.interpret(inputs=fgadr_tensors.to(DEVICE),
#                                        experimental_sets=[[smallDots_concept,softEx_concept,hemo_concept,hardEx_concept]],
#                                        target=4 #The target class (DR level 4)
#                                        #n_steps=5, #This is for the LayerIntegratedGradients: steps used by the approx. method (default is 50)
#                                       )

print(scores)
print('Finished with interpretation!')


########## Code for plotting ################
#Boxplot for significance testing:
n = 5
def show_boxplots(layer,layerstring, metric='sign_count'):

    def format_label_text(experimental_sets):
        concept_id_list = [exp.name if i == 0 else \
                             exp.name.split('/')[-1][:-1] for i, exp in enumerate(experimental_sets[0])]
        return concept_id_list

    n_plots = 2 #Plot NV vs negative concepts + negative vs negative

    fig, ax = plt.subplots(1, n_plots, figsize = (25, 7 * 1))
    fs = 18
    for i in range(n_plots):
        esl = experimental_sets[i * n : (i+1) * n]
        P1, P2, pval, relation = get_pval(scores, esl, layer, metric)

        ax[i].set_ylim([0, 1])
        #Andrea: added if/else for compatibility with densenet:
        if len(layerstring)>1:
            ax[i].set_title(layerstring + "-" + metric + " (pval=" + str(pval) + " - " + relation + ")", fontsize=fs)
        else:
            ax[i].set_title(layer + "-" + metric + " (pval=" + str(pval) + " - " + relation + ")", fontsize=fs)
        ax[i].boxplot([P1, P2], showfliers=True)

        ax[i].set_xticklabels(format_label_text(esl), fontsize=fs)

    #plt.show()
    print('Saving boxplot to file:')
    plt.savefig('NV_Denseblock4_BoxplotClass2_Densenet121_100epochsRepresentative.png', bbox_inches = 'tight')


def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)
        pos = [np.arange(len([model_features._modules['denseblock2'],
model_features._modules['denseblock3'],model_features._modules['denseblock4']]))]
        #pos = [np.arange(len(['Mixed_6d','Mixed_6e','Mixed_7c']))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len([model_features._modules['denseblock2'],
model_features._modules['denseblock3'],model_features._modules['denseblock4']]))])
        #_ax.set_xticks([r + 0.3 * barWidth for r in range(len(['Mixed_6d','Mixed_6e','Mixed_7c']))])
        _ax.set_xticklabels(['denseblock2','denseblock3','denseblock4'], fontsize=16, rotation = 45)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    print('Saving concept plot to file:')
    plt.savefig('CompareAllConceptClass4_Densenet121_100epochs_positivesRepresentative.png', bbox_inches = 'tight')
#Plot TCAV scores for the experimental set:
#plot_tcav_scores(experimental_sets[:5], scores)
plot_tcav_scores(experimental_sets, scores)

#Plot the boxplot for layer Mixed_7c:
#show_boxplots(layer='Mixed_7c',layerstring='')
#show_boxplots(layer = model_features._modules['denseblock4'],layerstring = 'denseblock4')