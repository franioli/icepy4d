import pandas as pd
import numpy as np

from typing import List


def read_data_to_df(
    file_path: str,
    delimiter: str = ',',
    header: int = 0,
    col_names: List[str] = None,
    index_col=None,
) -> pd.DataFrame:
    ''' Read text file to pandas dataframe
    '''
    df = pd.read_csv(file_path,
                     sep=delimiter,
                     header=header,
                     names=col_names,
                     index_col=index_col)
    return df
