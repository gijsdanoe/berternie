# BERT and ERNIE Framework
# Copyright (C) 2024 Gijs Danoe
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import config
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, MultiHeadAttention, Attention, Embedding
from tensorflow.keras.optimizers import Adam
import nltk


np.random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

nltk_data_path = config.nltk_data_path
nltk.data.path.append(nltk_data_path)

def open_data(data_path, data_vec_path):
    df = pd.read_csv(data_path, delimiter=',', on_bad_lines='warn', header=0, engine='python', encoding='utf-8',
                          encoding_errors='replace')
    preprocessed_data = np.load(data_vec_path)
    preprocessed_data = pd.DataFrame(preprocessed_data)

    return df, preprocessed_data

train_df, train_vec = open_data(config.train_df,config.train_vec)

# access all features separately for the model
train_emb = train_vec.iloc[:,:768]
train_icpc_enc = train_vec.iloc[:,768].to_numpy().reshape(-1,1)
train_mo_enc = train_vec.iloc[:,769:771]
train_age_scaled = train_vec.iloc[:,771].to_numpy().reshape(-1,1)

train_set = [train_emb, train_icpc_enc, train_mo_enc, train_age_scaled]

#### MODEL ####
with tf.device('/GPU:0'):
    embeddings_dim = train_emb.shape[1]
    icpc_dim = train_icpc_enc.shape[1]
    age_dim = train_age_scaled.shape[1]
    month_dim = train_mo_enc.shape[1]

    embeddings_input = Input(shape=(embeddings_dim,))
    icpc_input = Input(shape=(icpc_dim,))
    age_input = Input(shape=(age_dim,))
    month_input = Input(shape=(month_dim,))

    # embed ICPC
    icpc_emb = Embedding(input_dim=102, output_dim=20)(icpc_input)
    icpc_emb = tf.keras.layers.Flatten()(icpc_emb)

    # hierarchical attention
    combined_input1 = Concatenate()([embeddings_input, icpc_input])
    att1 = Attention()([combined_input1,combined_input1])

    combined_input2 = Concatenate()([att1, month_input, age_input])
    att = Attention()([combined_input2,combined_input2])


    x = Dense(256, activation='relu')(att)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)

    latent_space = Dense(32, activation='relu', name='latent_space')(x)

    x = Dense(64, activation='relu')(latent_space)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    decoded_output = Dense(embeddings_dim + icpc_dim + month_dim + age_dim, activation='linear', name='decoded_embedding')(x)


    autoencoder = Model(inputs=[embeddings_input, icpc_input, month_input, age_input], outputs=decoded_output)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    combined_train_input = np.concatenate(train_set, axis=1)
    history = autoencoder.fit(train_set, combined_train_input,
                              epochs=5,
                              batch_size=32,
                              shuffle=True,
                              validation_split=0.2)

    autoencoder.save(config.autoencoder)