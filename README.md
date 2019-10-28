# ML Project 1
## Description of the project
The code is clearly separated into 4 different parts.
- Exploratory data analysis
  - Correlation Matrix
  - PCA
  - Missing values
   - Replacement by mean
   - Separation into patterns
  - Distributions of features
   - Histograms
   - Boxplots
- Feature processing
  - Interactions between variables
- Implementations
  - Least Squares Gradient Descent (GD)
  - Least Squares SGD
  - Least Squares normal equations
  - Ridge regression
  - Logistic regression
        - GD
        - Newton
        - Regularized
- More complex analysis
  - Tuning hyperparameters
- Generating predictions

### Exploratory data analysis
We take a closer look at the data and try to identify patterns and features which are the most important for classification later on.
We replace the missing values by the mean over the columns or even take them out entirely for all the plots showed in order to
get a true sense of their distribution.

### Feature processing
We normalize the data. We also add an offset column, we augment the features up to a certain degree and 
we split the data into test and train sets for cross-validation.

### Implementations
We implement all the Machine Learning algorithms seen in the course. Each time it is relevant,
we also run them on the polynomial version of the data. When there is a hyperparameter to choose such as 
the ridge regression lambda term, we run the algorithm for a range of possible hyperparameters and evaluate the 
Root Mean Squared Error on the train and test set separately, to check for under or overfitting.

### More complex analysis
Here, we separate the test data into 6 different subgroups according to the patterns of missing values that we found in the data. 
We of course verified that the same patterns were present in the test set so that we ensure that our analysis is relevant.
For eahc subgroup, we perform Logistic regression. We condider the following hyperparameters: degree of feature expansion and lambda (ridge term).  
We tune the hyperparameters of each subgroup separately. We choose the right hyperparameters thanks to numerical (minimum loss value)
and graphical analysis (Boxplots). We also do cross-validation for each subgroup.
We end up with a 6 different models, one for each subgroup.

### Generating predictions
We generate predictions on the given test with the weights obtained using the provided functions.

## How to run the code
In order to be able to run the code, you should be able to run Jupyter Notebooks (requires the installation of Anaconda), 
and the following libraries should be installed: numpy and matplotlib.  
Notice that you should first download the data to be able to run the code, they are available at AICrowd at the EPFL Machine Learning Higgs 2019 challenge, in the tab 'resources': https://www.aicrowd.com/challenges/epfl-machine-learning-higgs-2019/dataset_files.  
The code runs sequentially, so it is very important to start running the code from the begining of the file.  
Also, the tuning of the hyperparameters is a lengthy process (30min) so we stored the results in a numpy file. So you can just skip this cell and run the next one where the data from the file is loaded.

## Folder plan 
You should put the downoaded files in a new folder 'data' in the folder 'project1'. 
The folders are organized this way : 
- project1
    - project1_description.pdf
    - data
    - scripts
       - src
       - utilities
       - plots
         
 The project to run is in the following path : /project1/scripts/src/project1.ipynib.  
 You can find all the requested regression methods in the path : /project1/scripts/utilities/implementations.py.  
 The other files in the folder utilities are all the functions we used to modularize the code in the project.  
 The plots folder contains all the savec plots produced by the project1 script.  
 The src folder contains all the main notebook where all the listed mehtods are used.  
 
 
        
