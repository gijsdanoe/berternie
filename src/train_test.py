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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, normalize, LabelEncoder
import numpy as np

np.random.seed(config.random_seed)
tf.random.set_seed(config.random_seed)

def open_data(data_path, data_vec_path):
    init_df = pd.read_csv(data_path, delimiter=',', on_bad_lines='warn', header=0, engine='python', encoding='utf-8',
                          encoding_errors='replace')
    init_df.rename(columns={'combined_so': 'soep_text', 'combined_soep': 'soep_text'}, inplace=True)
    columns = ['soep_text','gp_note_icpc','sex', 'age_at_consult', 'postal_code', 'ab_treatment','gp_note_month', 'gp_note_year', 'age_cat']
    df = init_df[columns]

    if config.english:
        df['sex'] = df['sex'].replace('V', 'F')

    preprocessed_data = np.load(data_vec_path, allow_pickle=True)
    preprocessed_data = pd.DataFrame(preprocessed_data)

    return df, preprocessed_data



df, preprocessed_data = open_data(config.ahon_df, config.ahon_vec)
df_mumc, preprocessed_data_mumc = open_data(config.mumc_df, config.mumc_vec)
df_rb, preprocessed_data_rb =  open_data(config.rb_df, config.rb_vec)


test_start_month = config.test_start_month
test_start_year = config.test_start_year
test_end_month = config.test_end_month
test_end_year = config.test_end_year

train_df = df[(df['gp_note_year'] < test_start_year) | ((df['gp_note_year'] == test_start_year) & (df['gp_note_month'] < test_start_month))]
train_indices = train_df.index
train_vec = preprocessed_data.iloc[train_indices]

if config.test_on == 'ahon':
    test_df = df[((df['gp_note_year'] > test_start_year) | ((df['gp_note_year'] == test_start_year) & (df['gp_note_month'] >= test_start_month))) & ((df['gp_note_year'] < test_end_year) | ((df['gp_note_year'] == test_end_year) & (df['gp_note_month'] < test_end_month)))]
    test_indices = test_df.index
    test_vec = preprocessed_data.iloc[test_indices]

# external validation
elif config.test_on == 'mumc':
    test_df = df_mumc[((df_mumc['gp_note_year'] > test_start_year) | ((df_mumc['gp_note_year'] == test_start_year) & (df_mumc['gp_note_month'] >= test_start_month))) & ((df_mumc['gp_note_year'] < test_end_year) | ((df_mumc['gp_note_year'] == test_end_year) & (df_mumc['gp_note_month'] < test_end_month)))]
    test_indices = test_df.index
    test_vec = preprocessed_data_mumc.iloc[test_indices]

elif config.test_on == 'rb':
    test_df = df_rb[((df_rb['gp_note_year'] > test_start_year) | ((df_rb['gp_note_year'] == test_start_year) & (df_rb['gp_note_month'] >= test_start_month))) & ((df_rb['gp_note_year'] < test_end_year) | ((df_rb['gp_note_year'] == test_end_year) & (df_rb['gp_note_month'] < test_end_month)))]
    test_indices = test_df.index
    test_vec = preprocessed_data_rb.iloc[test_indices]

#ADD ARTIFICIAL DATA
if config.artificial_data:
    normal = config.westnile_df
    vectorized = config.westnile_vec
    artificial_df, artificial_vec = open_data(normal, vectorized)
    test_df, test_vec = pd.concat([test_df, artificial_df], axis=0), pd.concat([test_vec, artificial_vec], axis=0)

train_emb = train_vec.iloc[:,:768]
test_emb = test_vec.iloc[:,:768]

scaler = StandardScaler()
train_age_scaled, test_age_scaled = scaler.fit_transform(train_df[['age_at_consult']]), scaler.transform(test_df[['age_at_consult']])


def sinusoidal_encoding(series, period=12):
    sin_values = np.sin(2 * np.pi * series / period)
    cos_values = np.cos(2 * np.pi * series / period)

    return np.column_stack((sin_values, cos_values))


train_mo_enc, test_mo_enc = sinusoidal_encoding(train_df['gp_note_month']), sinusoidal_encoding(test_df['gp_note_month'])

unique_labels = df['gp_note_icpc'].dropna().unique()
label_mapping = {val: idx for idx,val in enumerate(sorted(unique_labels))}
train_icpc_enc, test_icpc_enc = train_df['gp_note_icpc'].map(label_mapping).fillna(-1).to_numpy().reshape(-1,1), test_df['gp_note_icpc'].map(label_mapping).fillna(-1).to_numpy().reshape(-1,1)

train_set = [train_emb, train_icpc_enc, train_mo_enc, train_age_scaled]
combined_train_input = np.concatenate(train_set, axis=1)
np.save(config.train_vec,combined_train_input)
train_df.to_csv(config.train_df, index=False)

test_set = [test_emb, test_icpc_enc, test_mo_enc, test_age_scaled]
combined_test_input = np.concatenate(test_set, axis=1)
np.save(config.test_vec,combined_test_input)
test_df.to_csv(config.test_df, index=False)

print('Train and test sets created successfully in /data/train_test/')