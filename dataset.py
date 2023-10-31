from typing import List, Dict
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


class Dataset:
    def __init__(self, df, r=None):
        self.df = df
        self.scope = df.columns.tolist()
        if r is None:
            self.r = dict(zip(self.scope, (1+df.max(axis=0)).tolist()))
        else:
            self.r = r

    def split(self, v: str) -> List:
        r = {key: value for key, value in self.r.items() if key != v}
        datasets = [
            Dataset(self.df[self.df[v] == value].drop([v], axis=1), r)
            for value in range(self.r[v])
        ]
        return datasets

    def subset(self, V: List[str]):
        r = {key: value for key, value in self.r.items() if key not in V}
        return Dataset(self.df[V], r)

    def counts(self, V: List[str]) -> Dict:
        return self.df[V].value_counts().to_dict()

    def __len__(self):
        return len(self.df)

    def as_df(self):
        return self.df

    def bootstrap_samples(self, k: int, random_states: List[int]):
        for i in range(k):
            sample = self.df.sample(
                frac=1, replace=True, random_state=random_states[i])
            yield Dataset(sample, r=self.r)


class GroupedDataset(Dataset):
    def __init__(self, df, r=None, group_col="group"):
        super().__init__(df, r)
        self.group_col = group_col
        self.scope = [each for each in self.scope if each != group_col]

    def split(self, v: str) -> List:
        r = {key: value for key, value in self.r.items() if key != v}
        datasets = [
            GroupedDataset(self.df[self.df[v] == value].drop([v], axis=1), r, self.group_col)
            for value in range(self.r[v])
        ]
        return datasets

    def subset(self, V: List[str]):
        r = {key: value for key, value in self.r.items() if key not in V}
        return GroupedDataset(self.df[V+[self.group_col]], r, self.group_col)

    def counts(self, V: List[str], by_group=False):
        if not by_group:
            return self.df[V].value_counts().to_dict()
        else:
            return [self.df.query(f"{self.group_col} == {group}")[V].value_counts().to_dict()
                    for group in range(self.r[self.group_col])]

    def as_df(self, with_group=False):
        if not with_group:
            return self.df.drop([self.group_col], axis=1)
        else:
            return self.df

    @property
    def datasets(self):
        r = {key: value for key, value in self.r.items() if key !=
             self.group_col}
        return [
            Dataset(self.df.query(f"{self.group_col} == {group}").drop([self.group_col], axis=1), r)
            for group in range(self.r[self.group_col])
        ]

    def bootstrap_samples(self, k: int, random_states: List[int]):
        for i in range(k):
            samples = [
                g.sample(frac=1, replace=True, random_state=random_states[i])
                for _, g in self.df.groupby(by=self.group_col)
            ]
            sample = pd.concat(samples, ignore_index=True, sort=False)
            # sample = self.df.sample(frac=1, replace=True, random_state=random_states[i])
            yield GroupedDataset(sample, r=self.r, group_col=self.group_col)


# 1: White
# 2: Black
# 3: Hispanic
# 5: Asian
def get_dataset(name: str):
    if name == "numom2b_gdm_race":
        data = pd.read_csv(path.join("raw-data", "Data_for_AR.csv"))
        select = [
            *[f'DM_hist{i}' for i in range(1, 8+1)],
            *[f'Race_{i}' for i in range(1, 9+1)],
            'HiBP1', 'HiBP2',
            'PCOS1', 'PCOS2',
            'Age_at_V1', 'BMI', 'METs',
            'GDM'
        ]
        data = data[select].copy()
        selected_races = data[["Race_1", "Race_2",
                               "Race_3", "Race_5"]].any(axis=1)
        data = data[selected_races].dropna()

        data["Hist"] = (
            data[[f'DM_hist{i}' for i in range(1, 8+1)]] == 1).any(axis=1).astype('int')
        data["HiBP"] = (
            data[[f'HiBP{i}' for i in [1, 2]]] == 1).any(
            axis=1).astype('int')
        data["PCOS"] = (
            data[[f'PCOS{i}' for i in [1, 2]]] == 1).any(
            axis=1).astype('int')
        data["Race"] = pd.from_dummies(
            data[[f'Race_{i}' for i in [1, 2, 3, 5]]],
            sep="_").Race.astype(int).tolist()
        data.Race.replace({1: 0, 2: 1, 3: 2, 5: 3}, inplace=True)

        data['Age'] = (data["Age_at_V1"] > 35).astype(int)
        data.drop(['Age_at_V1'], axis=1, inplace=True)

        data.BMI = (data.BMI > 30).astype(int)

        data.METs = (data.METs > 450).astype(int)

        data.drop([f'Race_{i}' for i in range(1, 9+1)], axis=1, inplace=True)
        data.drop([f'DM_hist{i}' for i in range(1, 8+1)],  axis=1, inplace=True)
        data.drop([f'HiBP{i}' for i in [1, 2]],  axis=1, inplace=True)
        data.drop([f'PCOS{i}' for i in [1, 2]],  axis=1, inplace=True)

        r = {column: data[column].max()+1 for column in data.columns}
        train, test = train_test_split(
            data, train_size=0.8, stratify=data[["Race", "GDM"]],
            random_state=0)
        return GroupedDataset(
            train, r, group_col='Race'), GroupedDataset(
            test, r, group_col='Race')

    elif name == "numom2b_ptb_race":
        data = pd.read_csv(path.join("raw-data", "Data_for_AR.csv"))

        select = [
            *[f'DM_hist{i}' for i in range(1, 8+1)],
            *[f'Race_{i}' for i in range(1, 9+1)],
            'HiBP1', 'HiBP2',
            'PCOS1', 'PCOS2',
            'Age_at_V1', 'BMI', 'METs',
            'GDM', 'PReEc', 'PTB',
        ]

        data = data[select].copy()
        selected_races = data[["Race_1", "Race_2",
                            "Race_3", "Race_5"]].any(axis=1)
        data = data[selected_races].dropna()
        data["Hist"] = (
            data[[f'DM_hist{i}' for i in range(1, 8+1)]] == 1).any(axis=1).astype('int')
        data["HiBP"] = (
            data[[f'HiBP{i}' for i in [1, 2]]] == 1).any(
            axis=1).astype('int')
        data["PCOS"] = (
            data[[f'PCOS{i}' for i in [1, 2]]] == 1).any(
            axis=1).astype('int')
        data["Race"] = pd.from_dummies(
            data[[f'Race_{i}' for i in [1, 2, 3, 5]]],
            sep="_").Race.astype(int).tolist()
        data.Race.replace({1: 0, 2: 1, 3: 2, 5: 3}, inplace=True)

        data['Age'] = (data["Age_at_V1"] > 18) & (data["Age_at_V1"] < 35)
        data.drop(['Age_at_V1'], axis=1, inplace=True)

        data['BMI'] = data['BMI'] > 30
        data.METs = (data.METs > 450).astype(int)
        data.drop([f'Race_{i}' for i in range(1, 9+1)], axis=1, inplace=True)
        data.drop([f'DM_hist{i}' for i in range(1, 8+1)],  axis=1, inplace=True)
        data.drop([f'HiBP{i}' for i in [1, 2]],  axis=1, inplace=True)
        data.drop([f'PCOS{i}' for i in [1, 2]],  axis=1, inplace=True)

        r = {column: data[column].max()+1 for column in data.columns}
        train, test = train_test_split(
            data, train_size=0.8, stratify=data[["Race", "PTB"]],
            random_state=0)
        return GroupedDataset(
            train, r, group_col='Race'), GroupedDataset(
            test, r, group_col='Race')

    else:
        raise ValueError("Invalid data set name")
