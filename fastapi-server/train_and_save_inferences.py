import json
import os
from argparse import ArgumentParser

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_preprocessor import DataPreprocessor


def save_inferences(
        model, encoder, scaler, selected_features, basedir: str = 'inferences'
):
    os.makedirs(basedir, exist_ok=True)
    joblib.dump(encoder, os.path.join(basedir, "encoder.pkl"))
    joblib.dump(scaler, os.path.join(basedir, "scaler.pkl"))
    joblib.dump(model, os.path.join(basedir, "model.pkl"))

    with open(os.path.join(basedir, 'selected_features.json'), 'w') as file:
        json.dump(selected_features, file)


def fit_encoder(data: DataFrame, categorical_features) -> OneHotEncoder:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(data[categorical_features])

    return encoder


def fit_scaler(data: DataFrame, numeric_features) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(data[numeric_features])

    return scaler


def train_model(data: DataFrame) -> RandomForestRegressor:
    target = data['cost']
    features = data.drop(columns=['cost'])

    model = RandomForestRegressor(
        random_state=42,
        max_depth=15,
        min_samples_leaf=4,
        min_samples_split=15,
        n_estimators=200,
        bootstrap=True,
    )
    model.fit(features, target)

    return model


def main():
    args = parse_args()

    data = pd.read_csv(args.train_data)

    fit_data = DataPreprocessor.convert_to_categories(data)

    categorical_features = fit_data.select_dtypes(include=['category']).columns
    numeric_features = fit_data.select_dtypes(include=['int', 'float']).columns
    numeric_features = numeric_features.drop('cost')

    encoder = fit_encoder(fit_data, categorical_features)
    scaler = fit_scaler(fit_data, numeric_features)

    preprocessor = DataPreprocessor(
        encoder=encoder,
        scaler=scaler,
    )
    data_preprocessed = preprocessor.preprocess(data)

    save_inferences(
        model=train_model(data_preprocessed),
        encoder=encoder,
        scaler=scaler,
        selected_features=[value for value in preprocessor.selected_features if value != 'cost'],
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--train-data", help="Path to file with train data",
                        default="train.csv")
    return parser.parse_args()


if __name__ == "__main__":
    main()
