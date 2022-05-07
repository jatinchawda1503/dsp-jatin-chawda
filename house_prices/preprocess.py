import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
import joblib

multiple_regression = joblib.load('../models/model.joblib', mmap_mode=None)
ohe = joblib.load('../models/one_hot_encoder.joblib', mmap_mode=None)
sc = joblib.load('../models/standard_scaler.joblib', mmap_mode=None)

def select_clean_data(data):
    X = data[['GarageArea', 'GarageCars', 'OverallQual','GrLivArea']]
    X = X.dropna().reset_index(drop=True)
    if 'SalePrice' in data.columns:
        y = data['SalePrice'].values.reshape(-1,1)
    else:
        return X 
    return X,y


def train_test_split_data(X_feature,y_feature,size,random):
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y_feature, test_size = size, random_state = random)
    return X_train,X_test, y_train, y_test


  
def encoding_with_one_hot_encoder(data,feature_1 ,feature_2):
    fitting_encoder = ohe.fit(data[[feature_1 , feature_2]]).transform(data[[feature_1 , feature_2]])
    col_names = ohe.get_feature_names_out(input_features = [feature_1 , feature_2])
    encoder_df = pd.DataFrame(fitting_encoder,columns=col_names,index = data.index)
    data = data.join(encoder_df)
    data = data.drop([feature_1,feature_2],axis=1)
    return data

def scaling_with_standard_scalar(data,feature_1,feature_2):
    data[[feature_1, feature_2]] = sc.fit(data[[feature_1, feature_2]]).transform(data[[feature_1, feature_2]])
    return data


def train_linear_regression(X_data,y_data):
    regression = multiple_regression.fit(X_data, y_data)
    return regression
    

def encoding_with_one_hot_encoder_with_transform(data,feature_1 ,feature_2):
    fitting_encoder = ohe.transform(data[[feature_1 , feature_2]])
    col_names = ohe.get_feature_names_out(input_features = [feature_1 , feature_2])
    encoder_df = pd.DataFrame(fitting_encoder,columns=col_names,index = data.index)
    data = data.join(encoder_df)
    data = data.drop([feature_1,feature_2],axis=1)
    return data

def scaling_with_standard_scalar_with_transform(data,feature_1,feature_2):
    data[[feature_1, feature_2]] = sc.transform(data[[feature_1, feature_2]])
    return data


def predict_data(data_to_predict):
    y_pred = multiple_regression.predict(data_to_predict)
    return y_pred


def compute_rmsle(y_val: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))
    return round(rmsle, precision)
