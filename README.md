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

### References
1. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., Sayres, R.: Interpretability beyond feature attribution: Quantitative testing with concept
activation vectors (TCAV). In: Dy, J., Krause, A. (eds.) Proceedings of the 35th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 80, pp. 2668–2677. PMLR (2018), https://proceedings.mlr.press/v80/kim18d.html
2. Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P.: Concept bottleneck models. In: III, H.D., Singh, A. (eds.) Proceedings of
the 37th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 119, pp. 5338–5348. PMLR (13–18 Jul 2020), https://proceedings.mlr.press/v119/koh20a.html
