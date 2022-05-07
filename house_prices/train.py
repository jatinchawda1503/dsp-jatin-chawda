import pandas as pd
from house_prices.preprocess import select_clean_data
from house_prices.preprocess import train_test_split_data
from house_prices.preprocess import encoding_with_one_hot_encoder
from house_prices.preprocess import scaling_with_standard_scalar
from house_prices.preprocess import train_linear_regression
from house_prices.preprocess import encoder_with_transform
from house_prices.preprocess import scaling_with_transform
from house_prices.preprocess import predict_data
from house_prices.preprocess import compute_rmsle


def build_model(data: pd.DataFrame) -> dict[str, str]:
    X, y = select_clean_data(data)
    X_train, X_test, y_train, y_test = train_test_split_data(X,
                                                             y, 0.25, 0)
    X_train = encoding_with_one_hot_encoder(X_train,
                                            'GarageCars', 'OverallQual')
    X_train = scaling_with_standard_scalar(X_train, 'GarageArea', 'GrLivArea')
    train_linear_regression(X_train, y_train)
    X_test = encoder_with_transform(X_test, 'GarageCars', 'OverallQual')
    X_test = scaling_with_transform(X_test, 'GarageArea', 'GrLivArea')
    y_predict = predict_data(X_test)
    compute_rmsle(y_test, y_predict)
    return {'rsme': compute_rmsle(y_test, y_predict)}
