#Tyler Tobin - 001105522 - FYP 1682
# Machine Learning training for road situation classification

#---------- Imports ----------

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
import scipy
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.metrics import precision_score, recall_score

#----------- Testing Code ----------
'''
# create custom dataset
X = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0], [1, 1, 1, 1, 2, 0], [0, 1, 0, 1, 0, 0], [1, 1, 0, 1, 2, 0]])
y = np.array([0, 0, 1, 0, 1])

# split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create SVC classifier
svc = SVC(kernel='linear')

# fit classifier to training data
svc.fit(X_train, y_train)
prediction = svc.predict([[0, 1, 0, 1, 0, 0]])
print(prediction[0])

# evaluate performance on testing data
score = svc.score(X_test, y_test)
print(f'Test accuracy: {score}')

'''


#-------- Dataset loading ---------
dataset = pd.read_csv("situations.csv")
#rng = np.random.default_rng(0)
#Xy_df = dataset.iloc[rng.permutation(len(dataset))].reset_index(drop=True)

#print(Xy_df)
features = ['cam_east', 'cam_west', 'mean_weight_east', 'mean_weight_west', 'total_vehicles', 'obstruction']
#target = dataset.

X_raw = dataset[features]
y_raw = dataset.target_situation

#------------- Pre-processing ------------

X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.15, shuffle=True, random_state=0)

X_train_num = X_train_raw.select_dtypes(include=np.number)
X_train_cat = X_train_raw.select_dtypes(exclude=np.number)

#print(X_train_cat_total_vehicles)

numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

numeric_imputer.fit(X_train_num)
categorical_imputer.fit(X_train_cat)

X_train_num_imp = numeric_imputer.transform(X_train_num)
X_train_cat_imp = categorical_imputer.transform(X_train_cat)
#print(X_train_cat_imp)
X_test_num = X_test_raw.select_dtypes(include=np.number)
X_test_cat = X_test_raw.select_dtypes(exclude=np.number)
X_test_num_imp = numeric_imputer.transform(X_test_num)
X_test_cat_imp = categorical_imputer.transform(X_test_cat)

X_train_cat_total_vehicles = X_train_cat_imp[:, 2].reshape(-1, 1)
X_test_cat_total_vehicles = X_test_cat_imp[:, 2].reshape(-1, 1)

#print(X_train_cat_total_vehicles)

#----
#label encoding the first two columns of categorical data

encoder1 = LabelEncoder()
encoder2 = LabelEncoder()
encoder3 = LabelEncoder()
encoder4 = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[['one', 'two', 'three']])
encoder1.fit(X_train_cat_imp[:, 0])
encoder2.fit(X_train_cat_imp[:, 1])
encoder3.fit(X_train_cat_imp[:, 3])
X_onehot_cat_train = encoder4.fit_transform(X_train_cat_total_vehicles)
X_onehot_cat_test = encoder4.transform(X_test_cat_total_vehicles)
#one_hot_array = one_hot_encoded.toarray()
#print(X_onehot_cat_train)

X_train_cat_imp[:, 0] = encoder1.transform(X_train_cat_imp[:, 0])
X_train_cat_imp[:, 1] = encoder2.transform(X_train_cat_imp[:, 1])
X_train_cat_imp[:, 3] = encoder3.transform(X_train_cat_imp[:, 3])
X_train_cat_imp = np.delete(X_train_cat_imp, 2, axis=1)

X_test_cat_imp[:, 0] = encoder1.transform(X_test_cat_imp[:, 0])
X_test_cat_imp[:, 1] = encoder2.transform(X_test_cat_imp[:, 1])
X_test_cat_imp[:, 3] = encoder3.transform(X_test_cat_imp[:, 3])
X_test_cat_imp = np.delete(X_test_cat_imp, 2, axis=1)

print(X_onehot_cat_train)

X_train_cat_imp = np.concatenate([X_train_cat_imp, X_onehot_cat_train], axis=1).astype(int)

X_test_cat_imp = np.concatenate([X_test_cat_imp, X_onehot_cat_test], axis=1).astype(int)

#print(X_train_cat_imp)
#print(one_hot_encoded)

#print(X_train_num_imp)
'''
# data binning tests
for dataset in X_train_num_imp:
    if dataset[0] <= 2500:
        dataset[0] = 1
    elif dataset[0] > 2500 and dataset[0] <= 7000
'''


'''
#UNCOMMENT TO SEE NORMALISED DATA SCORE
# Scaler Object
scaler = MinMaxScaler()
# Fit on the numeric training data
scaler.fit(X_train_num_imp)
# Transform the training and test data
X_train_num_sca = scaler.transform(X_train_num_imp)
X_test_num_sca = scaler.transform(X_test_num_imp)
'''
#print(X_train_cat_imp)
#print(X_train_num_sca)

#X_raw = np.array(Xy_df[dataset[features]])
#y = np.array(Xy_df[target])
np.set_printoptions(precision=2, suppress=True)

X_train = np.concatenate([X_train_cat_imp, X_train_num_imp], axis=1)
X_test = np.concatenate([X_test_cat_imp, X_test_num_imp], axis=1)

#----------- Model -----------

model = ExtraTreesClassifier(n_estimators=10)
#model = SVC(kernel='linear')
#model = LogisticRegression(random_state=0)
#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

#----------- Training -----------

# fit classifier to training data
model.fit(X_train, y_train)

#----------- PRediction and evaluation ----------

# get the start time
st = time.time()
test_prediction = [[1, 1, 1, 0, 0, 1, 2187, 3965]]
test_prediction2 = [[1, 0, 1, 1, 0, 0, 1589, 0]]
#prediction = model.predict(test_prediction)
prediction = model.predict(X_test)
ps = precision_score(y_test, prediction, average='weighted')
recall = recall_score(y_test, prediction, average='macro')
et = time.time()
elapsed_time = et - st
print(prediction[0])
print(elapsed_time, " - First Prediction took")
print('precision: ',ps)
print('recall: ',recall)

prediction = model.predict(test_prediction2)
print(prediction[0])
# evaluate performance on testing data
score = model.score(X_test, y_test)
print(f'Test accuracy: {score}')


