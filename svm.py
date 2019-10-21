import numpy as np
np.random.seed(37)
import random
import pandas as pd

from sklearn.svm import SVC
from sklearn import preprocessing
# Att: You're not allowed to use modules other than SVC in sklearn, i.e., model_selection.

# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features
col_names_x = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
             'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
             'hours-per-week', 'native-country']
col_names_y = ['label']

numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss',
                  'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'native-country']


# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables.
def load_data(csv_file_path):
    # your code here
    file = open(csv_file_path)
    df = pd.read_csv(csv_file_path,
        header=None,
        names=col_names_x+col_names_y
       )
    # drop the NaN
#     df = df.dropna(axis=0, how="any")

    y = df['label'].astype('category')
    y = y.replace([" <=50K", " >50K"],[0, 1])
    
    # dataset has '?' in it, convert these into NaN, then fill NaN with the most frequent value of column
    df = df.replace(' ?', np.nan)
    df = df.replace(' Holand-Netherlands', np.nan)
    x = df.drop('label', axis=1)
    x = x.fillna(x.mode().iloc[0])
    
    #implement one hot encoding for the categorial variables
    categorical_ = ['workclass','education','marital-status','occupation','relationship',
                'race','sex','native-country'
    ]
    x = pd.get_dummies(x, columns=categorical_)
    
    # drop the column that is not in training data
#     if csv_file_path == "salary.2Predict.csv":
#         x = x.drop('native-country_ Holand-Netherlands', axis=1)
#     print("Partial data\n", x.iloc[0:10, :])
    
    # Normalization
    x = preprocessing.StandardScaler().fit_transform(x)
#     min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#     x = min_max_scaler.fit_transform(x)
    return x, y.to_numpy()

def fold(x, y, i, nfolds):
    # your code
    width = int(x.shape[0]/nfolds)
    x_left, x_test, x_right =  np.split(x,[i*width,(i+1)*width],axis=0)
    y_left, y_test, y_right =  np.split(y,[i*width,(i+1)*width],axis=0)
    x_train = np.concatenate((x_left,x_right), axis = 0)
    y_train = np.concatenate((y_left,y_right), axis = 0)
    return x_train, y_train, x_test, y_test

def get_acc(y_pred,y):
    correct = 0.0
    for i in range(y.shape[0]):
        correct += (y_pred[i]==y[i])
    return float(correct/len(y))

# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(training_csv):
    # load data and preprocess from filename training_csv
    x, y = load_data(training_csv)
    # hard code hyperparameter configurations, an example:
    param_set = [
                 {'kernel': 'linear', 'C': 1, 'degree': 5},
                 {'kernel': 'linear', 'C': 3, 'degree': 5},
                 {'kernel': 'rbf', 'C': 1, 'degree': 1},
                 {'kernel': 'rbf', 'C': 1, 'degree': 2},
                 {'kernel': 'rbf', 'C': 3, 'degree': 2},
                 {'kernel': 'rbf', 'C': 5, 'degree': 2},
                 {'kernel': 'poly', 'C': 3, 'degree': 2},
                 {'kernel': 'poly', 'C': 3, 'degree': 3},
                 {'kernel': 'poly', 'C': 1, 'degree': 3},
                 {'kernel': 'poly', 'C': 1, 'degree': 2},
    ]
    best_score = 0.0
    accuracys = []
    for params in param_set:
        val_acc = 0.0
        train_acc = 0.0
#         print(acc)
#         model = SVC(kernel=params['kernel'], C = params['C'], degree = params['degree'])
        for i in range(3):
            x_train, y_train, x_test, y_test = fold(x,y,i,3)
            model = SVC(kernel=params['kernel'], C = params['C'], degree = params['degree'])
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            y_pred1 = model.predict(x_train)
            train_acc += get_acc(y_pred1, y_train)
            val_acc += get_acc(y_pred,y_test)
#             print(acc)
        val_acc /= 3
        train_acc /= 3
#         print(acc)
        print("Train acc: ",train_acc," ",params)
        print("Test acc: ",val_acc," ",params)
#         print(model)
        if val_acc > best_score:
            best_score = val_acc
            best_model = model
    # your code here
    # iterate over all hyperparameter configurations
    # perform 3 FOLD cross validation
    # print cv scores for every hyperparameter and include in pdf report
    # select best hyperparameter from cv scores, retrain model 
    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(test_csv, trained_model):
    x_test, _ = load_data(test_csv)
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format 
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            if pred == 0:
                f.write('<=50K\n')
            else:
                f.write('>50K\n')



if __name__ == '__main__':
    training_csv = "salary.labeled.csv"
    testing_csv = "salary.2Predict.csv"
    # fill in train_and_select_model(training_csv) to 
    # return a trained model with best hyperparameter from 3-FOLD 
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter. 
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(training_csv)

    print("The best model was scored %.2f" % cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(testing_csv, trained_model)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.