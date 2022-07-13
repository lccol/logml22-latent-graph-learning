import argparse
from base64 import encode
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from models.rnn import LSTMBaseline
from models.lit_rnn import LitRNN
from utils import encode_labels

from torch import nn
from torch.optim import Adam
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
    parser.add_argument('--mlp-layer', type=str, dest='mlp_layer_str', help='String containing the hidden features of the final MLP classifier, comma-separated. The number of layers will be the number of elements.')
    parser.add_argument('--lr', type=float, help='Learning rate for Adam optimizer.')
    parser.add_argument('--wd', type=float, help='Value used for weight decay.')

    return parser.parse_args()

def parse_train_valid_test_splits(split_str: str) -> Tuple[float, float, float]:
    fields = split_str.split(',')
    assert 2 <= len(fields) <= 3
    tmp = [int(x) for x in fields]
    total = sum(tmp)
    res = [x / total for x in tmp]
    return tuple(res)

def parse_mlp_string(mlp_str: str) -> List[int]:
    fields = mlp_str.split(',')
    return [int(x) for x in fields]

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
    input_features = 12
    hidden_size = 64
    num_layers = 4
    bidirectional = True
    num_target_classes = 11
    target_column = 'Rhythm'
    pl.seed_everything(seed)

    args = parse()
    output_model_path = args.output_model_path
    output_performace = args.output_perf_path
    epochs = args.epochs
    batch_size = args.batch_size
    checkpoint_path = args.checkpoint_path
    split_str = args.split
    ts_duration = args.ts_duration
    mlp_str = args.mlp_str
    lr = args.lr
    wd = args.wd

    mlp_layer = parse_mlp_string(mlp_str)

    diag_df = pd.read_excel(diag_path)
    label_mapper = encode_labels(diag_df[target_column])
    print(f'Label mapper: {label_mapper}')

    train, validation, test = parse_train_valid_test_splits(split_str)
    train_files, validation_files, test_files = split_files(diag_df, [train, validation, test], seed)

    train_ecg = ECGDataset(basepath / 'ECGDataDenoised',
                    diag_path,
                    target_class_mapper=label_mapper,
                    target_column=target_column,
                    ts_duration=1000,
                    ignore_invalid_splits=True,
                    file_set=train_files)

    validation_ecg = ECGDataset(basepath / 'ECGDataDenoised',
                diag_path,
                target_class_mapper=label_mapper,
                target_column=target_column,
                ts_duration=1000,
                ignore_invalid_splits=True,
                file_set=validation_files)

    test_ecg = ECGDataset(basepath / 'ECGDataDenoised',
                diag_path,
                target_class_mapper=label_mapper,
                target_column=target_column,
                ts_duration=1000,
                ignore_invalid_splits=True,
                file_set=test_files)

    train_loader = DataLoader(train_ecg, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_loader = DataLoader(validation_ecg, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ecg, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LSTMBaseline(input_features,
                        hidden_size,
                        num_layers,
                        nclasses=len(label_mapper),
                        bidirectional=bidirectional,
                        mlp_layers=mlp_layer)

    optimizer_kls = Adam
    optimizer_args = {
        'lr': lr,
        'weight_decay': wd
    }
    loss = nn.BCELoss

    lit_model = LitRNN(model, loss, optimizer_kls, optimizer_args)
    trainer = pl.Trainer(max_epochs=epochs, check_val_every_n_epoch=10, accelerator='auto', gpus=1)

    trainer.fit(lit_model, train_loader, validation_loader)
    torch.save(lit_model.model, output_model_path)