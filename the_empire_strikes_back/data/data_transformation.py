from bisect import bisect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from the_empire_strikes_back.config.fitness import (
    MARKET,
    TIMEFRAMES,
)
from the_empire_strikes_back.config.general import (
    SEQUENCE_WIDTH,
    SEQUENCE_HEIGHT,
)


def load_transform_save() -> None:
    """ Transforms data into sequences and saves numpy arrays in files. """
    data_tf = {}
    for timeframe in TIMEFRAMES:
        filename = make_filename(MARKET, timeframe)
        data_tf[timeframe] = pd.read_parquet(f'data/{filename}')[:2500]

    data = np.zeros(
        (
            len(data_tf[TIMEFRAMES[0]])-SEQUENCE_WIDTH*12,
            SEQUENCE_HEIGHT,
            SEQUENCE_WIDTH,
            2,
        )
    )
    for idx, i in enumerate(
            range(SEQUENCE_WIDTH*12, len(data_tf[TIMEFRAMES[0]]))
    ):
        data_sequences = np.zeros(
            (
                SEQUENCE_HEIGHT,
                SEQUENCE_WIDTH,
                2,
            )
        )
        data_frame_first_tf = data_tf[TIMEFRAMES[0]].iloc[
            i - SEQUENCE_WIDTH:i, :
        ]
        converted_frame_first_tf = convert_frame(data_frame_first_tf)
        data_sequences[:, :, 0] = converted_frame_first_tf
        '''
        plt.plot(data_frame_first_tf['l'])
        plt.plot(data_frame_first_tf['h'])
        plt.show()
        plt.imshow(converted_frame_first_tf)
        plt.show()
        '''
        last_date = data_tf[TIMEFRAMES[0]]['datetime'].iloc[i]
        df_second_tf = data_tf[TIMEFRAMES[1]][
            data_tf[TIMEFRAMES[1]]['datetime'] < last_date
            ]
        df_second_tf = df_second_tf.iloc[-SEQUENCE_WIDTH:, :]
        converted_frame_second_tf = convert_frame(df_second_tf)
        data_sequences[:, :, 1] = converted_frame_second_tf
        data[idx] = data_sequences

    np.save('data/data.npy', data)


def convert_frame(
    data_frame: pd.core.frame.DataFrame
) -> np.ndarray:
    """ Converts frame to numpy array of 2D sequence. """
    converted_frame = np.zeros((SEQUENCE_HEIGHT, SEQUENCE_WIDTH))
    for i in range(0, len(data_frame)):
        current_high = data_frame['h'].iloc[i]
        current_low = data_frame['l'].iloc[i]
        position_high = find_position(data_frame, current_high)
        position_low = find_position(data_frame, current_low)

        if position_low != position_high:
            for p in range(position_low, position_high):
                converted_frame[SEQUENCE_HEIGHT-p, i] = 1

        else:
            converted_frame[SEQUENCE_HEIGHT-position_high, i] = 1

    return converted_frame


def find_position(
    data_frame: pd.core.frame.DataFrame,
    price: float
) -> int:
    """ Finds position of price in frame for given number of
    bisections. """
    min_price = data_frame['l'].min()
    max_price = data_frame['h'].max()
    segment = (max_price - min_price) / SEQUENCE_HEIGHT
    segments = []
    for i in range(SEQUENCE_HEIGHT+1):
        segments.append(min_price + segment*i)

    index = bisect(segments, price)
    return index


def make_filename(market: str, timeframe: str) -> str:
    """ Returns string with filename to load. """
    return f'{market}_{timeframe}.parquet'
