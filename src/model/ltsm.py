import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Input

"""
Overview of the model, this is mainly for refrence as this
model code lives in the vertex ai pipeline.
"""

features = [
    'made_cut', 'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 
    'sg_t2g', 'sg_total', 'avg_strokes_per_round', 
    'sg_putt_rolling_mean', 'sg_arg_rolling_mean', 
    'sg_app_rolling_mean', 'sg_ott_rolling_mean', 
    'sg_t2g_rolling_mean', 'sg_total_rolling_mean', 
    'pos_normalized'
]
sequence_length = 5
# Define model
model = Sequential([
    Input(shape=(sequence_length, len(features))),
    Masking(mask_value=0.0),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(1, activation='linear')
])
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)