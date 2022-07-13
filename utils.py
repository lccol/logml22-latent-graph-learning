import numpy as np
import pandas as pd

from sklearn import preprocessing
from typing import Union, List, Tuple, Optional, Dict

def encode_labels(label_column: Union[pd.Series, np.ndarray, List[str]]) -> Dict[str, int]:
    le = preprocessing.LabelEncoder()
    le.fit(label_column)

    classes = le.classes_
    inverse = le.inverse_transform(classes)

    return {inv: kls for inv, kls in zip(inverse, classes)}