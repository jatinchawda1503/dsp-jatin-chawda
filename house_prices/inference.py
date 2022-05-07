import pandas as pd
import numpy as np
from house_prices.preprocess import select_clean_data,train_test_split_data,encoding_with_one_hot_encoder,scaling_with_standard_scalar,train_linear_regression,encoding_with_one_hot_encoder_with_transform,scaling_with_standard_scalar_with_transform,predict_data,compute_rmsle

def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    X = select_clean_data(input_data)
    X = encoding_with_one_hot_encoder_with_transform(X,'GarageCars','OverallQual')
    X = scaling_with_standard_scalar_with_transform(X,'GarageArea','GrLivArea')
    pred_data = predict_data(X)
    return pred_data