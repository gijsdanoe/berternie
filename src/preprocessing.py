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

import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
from tqdm import tqdm
import config


# open data file
data_path = config.data_path_ahon
#data_path = config.data_path_rb
#data_path = config.data_path_mumc
#data_path = r'data/artificial data/westnile.csv'

init_df = pd.read_csv(data_path, delimiter=';', on_bad_lines='warn', header=0, engine='python', encoding='utf-8',
                      encoding_errors='replace')
init_df = init_df.drop_duplicates(subset=['contact_id','soep_code'])

df_pivot = init_df.pivot(index='contact_id', columns='soep_code',values='soep_text')
demo_columns = ['contact_id', 'sex','age_at_consult', 'gp_note_month','gp_note_year', 'postal_code','gp_note_icpc']
demo_df = init_df[demo_columns].drop_duplicates()
df_merged = pd.merge(df_pivot, demo_df, on='contact_id')
df_filtered = df_merged[df_merged[['S', 'O', 'E', 'P']].isna().sum(axis=1) <= 2]
df_filtered = df_filtered.fillna('')

df_filtered['combined_soep'] = df_filtered['S'] + ' ' + df_filtered['O'] + ' ' + df_filtered['E'] + ' ' + df_filtered['P']

# ANTIBIOTIC TREATMENT
# ab_regex = 'Al+b+en+d+|Al+b+en+d+azol+|Am+ox+i|Am+ox+i/Cl+av+|Am+ox+ic+il+in+e|Am+ox+ic+il+in+e/Cl+av+ul+aan+z+uur+|Aug+m+en+(th?)+in+|Av+el+ox+|Az+i(th?)+ro|Az+i(th?)+r+om+yc+in+e|B+ac+(th?)+r+im+el+|Br+ox+il+|C+ip+r+o|C+ip+r+o(f|ph)+l+ox+ac+in+e|C+ip+r+ox+in+|Cl+am+ox+yl+|Cl+ar+i(th?)+r+o|Cl+ar+i(th?)+r+om+yc+in+e|Cl+in+d+a|Cl+in+d+am+yc+in+e|Co(th?)+r+im+|Co(th?)+r+im+ox+azol+|D+al+ac+in+|D+ox+y+|D+ox+y+c+yc+l+in+e|Er+y+(th?)+r+oc+in+e|Er+y+(th?)+r+o|Er+y+(th?)+r+om+yc+in+e|Esk+azol+e|(f|ph)+en+e(th?)+|(f|ph)+en+e(th?)+ic+il+in+e|(f|ph)+en+ox+y+|(f|ph)+en+ox+y+me+(th?)+yl+p+en+ic+il+in+e|(f|ph)+l+ag+y+l+|(f|ph)+l+ox+ap+en+|(f|ph)+l+uc+l+ox+ac+il+in+e|(f|ph)+os+(f|ph)+o|(f|ph)+os+(f|ph)+om+yc+in+e|(f|ph)+ur+ab+id+|(f|ph)+ur+ad+an+tin+e|Kl+ac+id+|L+ev+o|L+ev+o(f|ph)+l+ox+ac+in+e|M+eb+en+d+|M+eb+en+d+azol+|M+etr+o|M+etr+on+id+azol+|M+on+ur+il+|M+ox+i|M+ox+i(f|ph)+l+ox+ac+in+e|N+itr+o|N+itr+o(f|ph)+ur+an+(th?)+oïn+e|N+(f|ph)+l+ox+|N+(f|ph)+l+ox+ac+in+e|N+or+ox+in+|P+en+iV+|P+en+ic+il+in+eV+|T+av+an+ic+|(th?)+r+im+/S+ul+(f|ph)+a|V+erm+ox+|V+ib+r+am+yc+in+|Z+i(th?)+r+om+ax+'
# df_filtered['ab_treatment'] = df_filtered['P'].str.contains(ab_regex, regex=True, na=False, flags=re.IGNORECASE).astype(int)

df_filtered = df_filtered.drop_duplicates(subset='contact_id')
filtered_texts_df = df_filtered[['combined_soep','gp_note_icpc', 'sex', 'age_at_consult','gp_note_month','ab_treatment', 'gp_note_year','postal_code']]
pd.set_option('display.max_columns', None)


#KEYWORD FILTERING
# with open('keywords.txt', 'r') as file:
#     symptom_list = [line.strip() for line in file]
# pattern = '|'.join(symptom_list)
#
# def check_text(row):
#     soep_text = row['combined_soep']
#
#     return pd.Series(soep_text).str.contains(pattern, case=False, na=False).iloc[0]
#
#
# mask = df_final.apply(check_text, axis=1)
# filtered_out = df_final.loc[~mask]
# filtered_texts_df = df_final.loc[mask]

# CLEAN M/V
filtered_texts_df['sex'] = filtered_texts_df['sex'].str.upper()

# age to float
filtered_texts_df['age_at_consult'] = pd.to_numeric(filtered_texts_df['age_at_consult'])

# age category
filtered_texts_df['age_cat'] = pd.cut(filtered_texts_df['age_at_consult'],
                       bins=[-1,0,3,5,12,17,25,40,55,70,float('inf')],
                       labels=['0','1-3','4-5','6-12','13-17','18-25','26-40','41-55','56-70','71+'])

text_features = ['combined_soep']


class BERTVectorizer:
    def __init__(self, model_name=config.bertje):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        embeddings = []
        for text in tqdm(X.values, desc='Processing texts'):
            text = str(text)
            inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)
        result = np.vstack(embeddings)
        print(f"Processed {len(embeddings)} texts into embeddings with shape: {result.shape}")
        return result

text_transformer = Pipeline(steps=[
    ('bert', BERTVectorizer())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, text_features)
    ]
)

preprocessed_data = preprocessor.fit_transform(filtered_texts_df)

filename = os.path.splitext(os.path.basename(data_path))[0]
np.save(f'data/{filename}_2na_vec', preprocessed_data)
filtered_texts_df.to_csv(f'data/{filename}_2na.csv', index=False)