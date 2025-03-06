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

# config

# variables
random_seed = 420
english = True
artificial_data = True

# ahon, rb, or mumc
test_on = 'ahon'

test_start_month = 2
test_start_year = 2020
test_end_month = 3
test_end_year = 2020

# paths

# paths to raw data for preprocessing
data_path_ahon = r'data/raw/journals_processed_ahon.csv'
data_path_rb = r'data/raw/BERTERNIE/journals_processed_radboud.csv'
data_path_mumc = r'data/raw/BERTERNIE/journals_processed_mumc.csv'

# paths to preprocessed data
ahon_df = r"data/preprocessed/journals_processed_ahon_2na.csv"
ahon_vec = r"data/preprocessed/journals_processed_ahon_2na_vec.npy"

rb_df = r"data/preprocessed/journals_processed_radboud_2na.csv"
rb_vec = r"data/preprocessed/journals_processed_radboud_2na_vec.npy"

mumc_df = r"data/preprocessed/journals_processed_mumc_2na.csv"
mumc_vec = r"data/preprocessed/journals_processed_mumc_2na_vec.npy"

westnile_df = r'data/westnile.csv'
westnile_vec = r'data/westnile.npy'

# paths to train and test sets
train_df = r'data/train_test/train_df.csv'
train_vec = r"data/train_test/train_set.npy"

test_df = r'data/train_test/test_df.csv'
test_vec = r"data/train_test/test_set.npy"

# model paths
autoencoder = r'models/autoencoder_model.h5'
bertje = r'models/bertje/'
opus = r'models/opus_nl_en/'

# other files
postcode3 = r'data/external/geo_postcodes3.csv'
stopwords = r'data/external/stopword_list'