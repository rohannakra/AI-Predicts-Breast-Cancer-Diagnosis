# AI Predicts Breast Cancer Diagnosis

Steps Taken:
* set up data
    * encode the target variables as 0 and 1 using ```pd.getdummies()```
    * split the data using ```train_test_split()```
* visualize the data
    * ```TSNE()``` makes the data 2D for visualizatin purposes
* create logistic regression model
    * train the model on the dataset
* visualize the results
    * use ```PCA()``` to make the data 2D (once again)
    * use ```coef_``` and ```intercept_``` attributes to show the decision boundary
