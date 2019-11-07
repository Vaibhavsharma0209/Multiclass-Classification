#importing
import pandas as pd
import numpy as np
from matplotlib import pyplot
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

#Cleaning
data = pd.read_csv('E:\Dataset credit card\development_dataset.csv')
data.info()
data.head()
data.drop(columns = 'VAR1', inplace = True)
data['VAR14'] = data['VAR14'].replace('.',np.NaN)
data.isna().sum()

#Too many nan values - if filled with other values then there will be some biasness in the data (dropped VAR17 and VAR9)
data.drop(columns = ['VAR17', 'VAR9'], inplace = True)
data = data.fillna(data.median())
data['VAR14'] = data['VAR14'].convert_objects(convert_numeric=True)


#splitting 
X = data.drop(columns = 'VAR21')
y = data["VAR21"]

#Encoder for y variable
enc =LabelEncoder()
y = enc.fit_transform(y)

#balancing the data
sm = SMOTE()
resampled_X, resampled_y = sm.fit_resample(X, y)

#splitting to train and test dataset
X_train, X_test, y_train, y_test = train_test_split(resampled_X, resampled_y, test_size = 0.2)


#XGBoost with depth of 15 and restriction level 7 
xgb = XGBClassifier(n_estimators=100,
                    max_depth=15, 
                    gamma = 7)
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["mlogloss"]
xgb.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True)
xgb.score(X_train, y_train)
xgb.score(X_test, y_test)


#grid search cv
#gs = GridSearchCV(xgb, param, cv=5, verbose = 3)
#param = {}
#gs.fit(X_train,y_train)
#gs.best_params_

#results of the mlogloss
results = xgb.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

#plotting the mlogloss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax.legend()
pyplot.ylabel('M Log Loss')
pyplot.title('XGBoost M Log Loss')
pyplot.show()


