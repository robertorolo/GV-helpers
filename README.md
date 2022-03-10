# GV-helpers

Helpers modules for isatis.neo.

## Instructions

Add the code below to your calculator initialization script.

    if app[ 'project_batch_path' ] not in sys.path: 
        sys.path.append(app[ 'project_batch_path'])

The import the desired module as. The file must be in the same folder as the batch file.

    import DDH_BH_analysis

### ca1_1_in_cat2

Plots bar graph given two categorical variables.

![cat1_in_cat2](/figs/KMeansLito.png)

### DDH_BH_analysys

Plots a DDH vs BH analysis given two arrays of twin samples.

![DDH_BH](/figs/P2O5_%25.png)

### outliers_helpers

Functions to help identifying outliers.

### PCA_analysis

Plots to help PCA analysis.

![PCA_analysis](/figs/PCA_analysis.png)

![PCA_circles](/figs/PCAcircle.png)

### SVC_smoothing_calibration

Reference: Cevik, I.S., Leuangthong, O., Caté, A. et al. On the Use of Machine Learning for Mineral Resource Classification. Mining, Metallurgy & Exploration 38, 2055–2073 (2021). https://doi.org/10.1007/s42461-021-00478-9

![SVC_calibration](/figs/SVCCalibUni.png)
