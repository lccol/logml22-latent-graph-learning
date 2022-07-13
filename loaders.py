import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from typing import Union, Dict, List, Tuple, Optional
from pathlib import Path

class ECGDataset(Dataset):
    def __init__(self, folder: Union[Path, str], diagnostic_path: Union[Path, str]) -> None:
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)
        if isinstance(diagnostic_path, str):
            diagnostic_path = Path(diagnostic_path)

        assert folder.is_dir()

        self.folder = folder
        self.diagnostic_path = diagnostic_path
        self.patient_data = pd.read_excel(diagnostic_path)
        self.ecg_list = self.read_ecgs()
        self.labels = self.patient_data['Rhythm'].values
        return

    def read_ecgs(self) -> List[np.ndarray]:
        res = []
        for _, row in self.patient_data.iterrows():
            filename = row['FileName'] + '.csv'
            fullpath = self.folder / filename
            assert fullpath.is_file()

            res.append(pd.read_csv(fullpath, header=None).values)
        return res
    
    def __len__(self) -> int:
        return len(self.ecg_list)

    def __get_item__(self, idx) -> Dict[str, np.ndarray]:
        return {'data': self.ecg_list[idx], 'label': self.labels[idx]}