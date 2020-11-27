#%%
import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%% Data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
np.random.shuffle(df_train.values)
stacked_train = np.column_stack((df_train['x'].values, df_train['y'].values))
stacked_test = np.column_stack((df_test['x'].values, df_test['y'].values))

#%% Gráfica
plt.plot(df_train['x'].where(df_train['color']==1).dropna(),
         df_train['y'].where(df_train['color']==1).dropna())
plt.plot(df_train['x'].where(df_train['color']==0).dropna(),
         df_train['y'].where(df_train['color']==0).dropna())
#%%
model = keras.Sequential([keras.layers.Dense(4, input_shape=(2,), activation='relu'),
                          keras.layers.Dense(2, activation='sigmoid')])

model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(stacked_train, df_train['color'].values, batch_size=4, epochs=5)
# %%
model.evaluate(stacked_test, df_test.color.values)