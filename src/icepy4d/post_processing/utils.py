from datetime import datetime, timedelta

from itertools import compress
from pathlib import Path
from typing import List, Tuple, Union


def find_closest_date_idx(
    datetime_list: List[datetime],
    date_to_find: datetime,
):
    closest = min(datetime_list, key=lambda sub: abs(sub - date_to_find))
    for idx, date in enumerate(datetime_list):
        if date == closest:
            return idx


def make_pairs(
    pcd_list: List[Path], step: int = 1, date_format: str = "%Y_%m_%d"
) -> dict:
    dt = timedelta(step)
    idx = pcd_list[0].stem.find("202")
    dates_str = [date.stem[idx:] for date in pcd_list]
    dates = [datetime.strptime(date, date_format) for date in dates_str]

    pair_dict = {}
    for i in range(len(pcd_list) - step):
        date_in = dates[i]
        date_f = date_in + dt
        idx_closest = find_closest_date_idx(dates, date_f)
        pair_dict[i] = (str(pcd_list[i]), str(pcd_list[idx_closest]))

    return (pair_dict, dates)
