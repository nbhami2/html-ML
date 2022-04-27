# html-ML
## Usage
Change csv_name to input data file, data.csv is merged data file with behavioral data. Machine learning techniques are applied to identify important features and correlations, build models, and predict behavioral data.

## Functions
25/75 test/train split is used. Each function is duplicated to assess data with and without sex taken into account. Regression is used due to the uncategorical nature of the behavioral data.

Feature Selction
------------
XGB Regressor
Gradient-boosted decision tree, uses parallel tree boosting. A weak tree is boosted using other weak trees to build an overall strong model. Gradient descent is used to minimize error with each iteration.

SelectKBest, f_regression
Finds k best features by eliminating all but k highest f values between feature and target.

SelectKBest, mutual_info_regression
Finds k best features by retaining k highest scores from features accounting for dependencies across k closest neighbors.

LinearRegression
Fits linear model with coeffecients B = B_0,B_1,...,B_n to minimize error between predicted and actual values. Error calculated using sum of least squares.

Model Generation/Prediction
------------
```
structural = ['fwhm','snr','cnr','fber','efc','qi1','qi2','icvs','rpve','inu','summary']
functional = ['efc','fber','fwhm','ghost_x','snr','dvars','gcor','mean_fd','num_fd','perc_fd','outlier','quality']
```
Generate train/test data using only structural/functional data.

RepeatedKFold - tuned features
Split data randomly into K folds and average models from each fold. Only uses k best featuers.

```
get_models() and evaluate_model()
```
Find which model (logistic regression, perceptron, decision tree regressor, random forest generatior, gradient boosting regressor) will provide best (least error) model.
