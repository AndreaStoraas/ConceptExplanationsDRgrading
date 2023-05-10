# Looking into Concept Explanation Methods for Diabetic Retinopathy Classification

This repo includes the source code for training the diabetic retinopathy (DR) classification models and generate explanations using Testing with Concept Activation Vectors (TCAV) [1] and Concept Bottleneck Models (CBMs) [2]. The Captum implementation of TCAV and the original source code for the CBMs were modified for the Densenet-121 architecture. All datasets are publicly available. For access to the datasets, please refer to the respective dataset sources. 

![My figure](./Figures/PlotCombined_ConceptCounts_TCAVscores_FGADRTestset_subplotsMayLong.png)

### Datasets
This work applies the following datasets:
* APTOS 2019: https://www.kaggle.com/competitions/aptos2019-blindness-detection
* Diabetic Retinopathy Detection: https://www.kaggle.com/competitions/diabetic-retinopathy-detection  
* Messidor-2: https://www.adcis.net/en/third-party/messidor2/
* FGADR: https://csyizhou.github.io/FGADR/
* DDR: https://github.com/nkicsl/DDR-dataset
* DIARETDB1: http://www.lut.it.fi/project/imageret
* IDRiD: https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid

### Guide for Running the Experiments
1. The datasets should be downloaded in a folder called Data in the root directory.
2. Run the code in the DataPreprocessing folder: 
  * First, sort and split the datasets into train, validation and test sets by running the respective Sort...-files. 
  * Second, crop out black areas from the images in the APTOS 2019, Diabetic Retinopathy Detection, FGADR, and Messidor-2 datasets for training the DR classification model for TCAV by running preprocessImages.py for each of the datasets. These cropped datasets should be saved in a CroppedDataKaggle folder in the Data folder. 
  * Finally, you can combine the cropped train, validation and test sets into larger train, validation and test sets by running combinedDatasets.py
3. For the **TCAV experiments**, run the code in the TCAV_experiments folder: 
  * Train a deep neural network using the Densenet-121 architecture for the TCAV experiments by running trainWeightedMulticlass.py. The final model should be saved in a folder called outputs in the root directory.
  * Test the resulting model for DR level classification by running testMulticlass.py.
  * Sort the concepts from the four datasets (FGADR, DDR, IDRiD and DiaretDB1) into concept folders by running SortConcepts... files. The sorted concepts are saved in concept folders for each dataset, as well as one concept folder combining the images from all four datasets.
  * Pick representative test sets from the larger DR level classification test set by running pickRepresentativeTest.py.
  * Get the TCAV scores for the representative test set by running TCAV_experimenting.py. NB! The mean and standard deviations must be saved manually!
  * Run plot_scoresSubplots.py to plot the TCAV scores for each DR level. 
  * Run count_conceptImages.py to get an overview of the concept distributions in the four concept datasets.
  * For exploring the effect of masking, create the masked concept images by running the code in the MaskingBySegmentations subfolder and then run TCAV_experimenting.py.
4. For the **Sequential Bottleneck experiments**, run the code in the SequentialBottleneck_experiments folder
  * Split the concept datasets (DDR, IDRiD, DiaretDB1) into train, validation and test sets for training the first part of the CBMs by running sortRawdataDDR_IDRiD_DiaretDB1.py.
  * Get overview files of the concepts for FGADR (all six concepts) and DDR, IDRiD, DiaretDB1 and FGADR (four concepts) by running SortConceptsFGADR.py and createOverviewDF_DDR_IDRiD_DiaretDB1.py. 
  * For training the first part of the CBMs, run trainBottleneckDensenet121.py (for all six concepts) and trainCombinedBottleneck.py (for four concepts). The final models should be saved in a folder called outputs in the root directory.
  * To extract the predicted concepts for training the second part of the CBMs, run getConceptPredictions.py. The predicted concepts should be saved in a subfolder called SequentialModelOutput.
  * For testing the model performances on predicting concepts, run testConceptClassifier.py (for all six concepts) and testCombined_ConceptClassifier.py (for four concepts). For plotting the predicted concepts, run PlotConceptPredictions_FGADRTestset.py
  * For training the second part of the CBMs for final DR level classifications, run train_Concept2DRLevel.py. 
  * For testing the second part of the CBMs, run test_Concept2DRLevel.py. 
  * For test time intervention (TTI), run the code in the subfolder called TestTimeIntervention.

### References
1. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., Sayres, R.: Interpretability beyond feature attribution: Quantitative testing with concept
activation vectors (TCAV). In: Dy, J., Krause, A. (eds.) Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 2668–2677. PMLR (2018), https://proceedings.mlr.press/v80/kim18d.html
2. Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P.: Concept bottleneck models. In: III, H.D., Singh, A. (eds.) Proceedings of
the 37th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 119, pp. 5338–5348. PMLR (13–18 Jul 2020), https://proceedings.mlr.press/v119/koh20a.html
