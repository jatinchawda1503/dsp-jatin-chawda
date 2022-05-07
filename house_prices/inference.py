import pandas as pd
import numpy as np
from house_prices.preprocess import select_clean_data
from house_prices.preprocess import encoder_with_transform
from house_prices.preprocess import scaling_with_transform
from house_prices.preprocess import predict_data


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    X = select_clean_data(input_data)
    X = encoder_with_transform(X, 'GarageCars', 'OverallQual')
    X = scaling_with_transform(X, 'GarageArea', 'GrLivArea')
    pred_data = predict_data(X)
    return pred_data
