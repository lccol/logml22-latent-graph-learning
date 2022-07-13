import argparse
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from dataset.loaders_old import ECGDataset
from pathlib import Path

from typing import Tuple, Dict, List, Union, Optional, Set

def parse():
    parser = argparse.ArgumentParser(description='Script to train a simple LSTM model')
    parser.add_argument('--output-model', type=str, dest='output_model_path', help='Path in which the trained torch model is exported.')
    parser.add_argument('--output-performance', type=str, dest='output_perf_path', help='Path in which the performance achieved on the test set is exported as csv file.')
    parser.add_argument('--epochs', type=int, help='Number of epochs used for training')
    parser.add_argument('--batch-size', type=int, dest='batch_size', help='Batch size.')
    parser.add_argument('--checkpoint-path', type=str, dst='checkpoint_path', help='Path to file used for checkpointing the model')
    parser.add_argument('--split', type=str, help='Train-Validation-Test split. Must be written in the form train,valid,test (e.g., 8,1,1)')
    parser.add_argument('--ts-duration', type=int, dest='ts_duration', help='Time-series duration in terms of number of samples. Used to split original time series into several smaller time series.')

    return parser.parse_args()

def parse_train_valid_test_splits(split_str: str) -> Tuple[float, float, float]:
    fields = split_str.split(',')
    assert 2 <= len(fields) <= 3
    tmp = [int(x) for x in fields]
    total = sum(tmp)
    res = [x / total for x in tmp]
    return tuple(res)

def split_files(df: pd.DataFrame,
                splits: List[Union[float, int]],
                seed: Optional[int]=False) -> Tuple[Set[str]]:
    res = []
    if isinstance(splits[0], float):
        assert sum(splits) == 1.0
    remaining = df
    selected = set()
    for split in splits:
        assert len(selected.intersection(set(df['FileName'].tolist()))) == 0
        sel = remaining.sample(frac=split, random_state=seed)
        sel_filenames = set(sel['FileName'].tolist())
        res.append(sel_filenames)
        remaining = remaining[~remaining['FileName'].isin(sel_filenames)]

    return tuple(res)

if __name__ == '__main__':
    basepath = Path.home() / 'datasets' / 'PhysioNet2020'
    diag_path = basepath / 'Diagnostics.xlsx'
    seed = 47
    pl.seed_everything(seed)

    args = parse()
    output_model_path = args.output_model_path
    output_performace = args.output_perf_path
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    split_str = args.split

    diag_df = pd.read_excel(diag_path)

    train, validation, test = parse_train_valid_test_splits(split_str)
    train_files, validation_files, test_files = split_files(diag_df, [train, validation, test], seed)

    train_ecg = ECGDataset(basepath / 'ECGDataDenoised',
                    diag_path,
                    ts_duration=1000,
                    ignore_invalid_splits=True,
                    file_set=train_files)

    validation_ecg = ECGDataset(basepath / 'ECGDataDenoised',
                diag_path,
                ts_duration=1000,
                ignore_invalid_splits=True,
                file_set=validation_files)

    test_ecg = ECGDataset(basepath / 'ECGDataDenoised',
                diag_path,
                ts_duration=1000,
                ignore_invalid_splits=True,
                file_set=test_files)

    train_loader = DataLoader(train_ecg, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_ecg, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ecg, batch_size=batch_size, shuffle=True, drop_last=False)