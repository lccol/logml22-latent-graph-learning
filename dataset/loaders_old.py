import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from typing import Union, Dict, List, Tuple, Optional, Set
from pathlib import Path

class ECGDataset(Dataset):
    def __init__(self, 
                folder: Union[Path, str], 
                diagnostic_path: Union[Path, str],
                target_class_mapper: Optional[Dict[str, int]]=None,
                target_column: str='Rhythm',
                ts_duration: Optional[int]=None,
                ignore_invalid_splits: Optional[bool]=False,
                file_set: Optional[Set[str]]=None,
                sample: Optional[Union[int, float]]=None,
                seed: Optional[int]=None) -> None:
        super().__init__()
        if isinstance(folder, str):
            folder = Path(folder)
        if isinstance(diagnostic_path, str):
            diagnostic_path = Path(diagnostic_path)

        assert folder.is_dir()

        self.folder = folder
        self.diagnostic_path = diagnostic_path
        self.target_class_mapper = target_class_mapper
        self.target_column = target_column
        self.ts_duration = ts_duration
        self.ignore_invalid_splits = ignore_invalid_splits # if True, ignore files for which df.shape[0] % ts_duration != 0
        self.file_set = file_set
        self.sample = sample # randomly samples just few files instead of all of them
        self.seed = seed

        self.patient_data = pd.read_excel(diagnostic_path)
        if not sample is None:
            if isinstance(sample, int):
                args = (sample, None)
            else:
                args = (None, sample)
            self.patient_data = self.patient_data.sample(*args, random_state=seed)

        if not target_class_mapper is None:
            self.patient_data[target_column] = self.patient_data[target_column].replace(target_class_mapper)
        if not self.file_set is None:
            self.patient_data = self.patient_data[self.patient_data['FileName'].isin(file_set)]
        self.ecg_list, self.labels = self.read_ecgs()
        return

    def read_ecgs(self) -> Tuple[List[np.ndarray], np.ndarray]:
        data = []
        labels = []
        for _, row in self.patient_data.iterrows():
            filename = row['FileName'] + '.csv'
            original_label = row[self.target_column]
            fullpath = self.folder / filename
            assert fullpath.is_file()

            X = pd.read_csv(fullpath, header=None).values
            if self.ts_duration is None:
                # simply append the entire numpy array as it is
                data.append(X)
                labels.append(original_label)
            else:
                if X.shape[0] % self.ts_duration != 0 and not self.ignore_invalid_splits:
                    raise ValueError(f'Error when reading {fullpath.stem} file: shape {X.shape} but ts_duration is {self.ts_duration}')
                    # assert X.shape[0] // self.ts_duration == 0 # check if the lines are a multiple of ts_duration
                if X.shape[0] % self.ts_duration != 0 and self.ignore_invalid_splits:
                    print(f'WARNING: ignoring file {fullpath.stem} because the number of rows ({X.shape[0]}) is not a multiple of {self.ts_duration}!')
                    continue
                sections = X.shape[0] // self.ts_duration

                data.extend(np.split(X, sections, axis=0))
                labels.extend([original_label for _ in range(sections)])

        return data, np.array(labels)
    
    def __len__(self) -> int:
        return len(self.ecg_list)

    def __get_item__(self, idx) -> Dict[str, np.ndarray]:
        return {'data': self.ecg_list[idx], 'label': self.labels[idx]}