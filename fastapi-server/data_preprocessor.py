import json
import os.path
from typing import List

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DataPreprocessor:
    scaler: StandardScaler
    encoder: OneHotEncoder
    selected_features: List[str] | None

    def __init__(self, scaler: StandardScaler, encoder: OneHotEncoder, selected_features: List[str] = None):
        self.scaler = scaler
        self.encoder = encoder
        self.selected_features = selected_features

    @classmethod
    def from_inferences(cls):
        scaler = joblib.load(os.path.join("inferences", "scaler.pkl"))
        encoder = joblib.load(os.path.join("inferences", "encoder.pkl"))
        with open(os.path.join("inferences", "selected_features.json"), "r") as f:
            selected_features = json.load(f)

        return cls(scaler, encoder, selected_features)

    @staticmethod
    def _remove_nan_and_inf(data: DataFrame) -> DataFrame:
        res_data = data.fillna(data.median())
        res_data = res_data.replace([np.inf, -np.inf], np.nan)

        return res_data

    @staticmethod
    def convert_to_categories(data: DataFrame) -> DataFrame:
        res_data = data.copy()

        res_data['unit_sales(in millions)'] = res_data['unit_sales(in millions)'].astype('category')
        res_data['total_children'] = res_data['total_children'].astype('category')
        res_data['num_children_at_home'] = res_data['num_children_at_home'].astype('category')
        res_data['avg_cars_at home(approx).1'] = res_data['avg_cars_at home(approx).1'].astype('category')

        return res_data

    def _encode_categorical(self, data: DataFrame, categorical_features: List[str]) -> DataFrame:
        res_data = data.copy()

        encoded_data = self.encoder.transform(res_data[categorical_features])
        encoded_df = pd.DataFrame(encoded_data, columns=self.encoder.get_feature_names_out(categorical_features))
        res_data = res_data.drop(columns=categorical_features).reset_index(drop=True)
        res_data = pd.concat([res_data, encoded_df], axis=1)

        return res_data

    def _scale_numerical(self, data: DataFrame, numeric_features: List[str]) -> DataFrame:
        res_data = data.copy()

        res_data[numeric_features] = self.scaler.transform(res_data[numeric_features])

        return res_data

    @staticmethod
    def select_features(data: DataFrame) -> List[str]:
        correlation_threshold = 0.00000001

        corr_with_target = data.corr()['cost'].abs().sort_values(ascending=False)
        target_correlation = corr_with_target.drop('cost')
        selected_features = target_correlation[
            target_correlation.abs() > correlation_threshold
            ].index.tolist()

        return selected_features + ['cost']

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        res_data = self._remove_nan_and_inf(data)
        res_data = self.convert_to_categories(res_data)

        categorical_features = res_data.select_dtypes(include=['category']).columns
        numeric_features = res_data.select_dtypes(include=['int', 'float']).columns
        if 'cost' in numeric_features:
            numeric_features = numeric_features.drop('cost')

        res_data = self._encode_categorical(res_data, categorical_features)
        res_data = self._scale_numerical(res_data, numeric_features)

        if self.selected_features is None:
            self.selected_features = self.select_features(res_data)

        return res_data[self.selected_features]
