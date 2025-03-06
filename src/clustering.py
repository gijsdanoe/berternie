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
import numpy as np
from tensorflow.keras.models import Model, load_model
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
import geopandas as gpd
from shapely import wkt
import umap.umap_ as umap
from wordcloud import WordCloud
from sklearn.metrics import silhouette_samples
from translator import Translator
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import warnings

warnings.simplefilter('ignore', UserWarning)

with open(config.stopwords, 'r') as f:
    custom_stopwords = set(f.read().splitlines())

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

test_df, test_vec = open_data(config.test_df,config.test_vec)

# access all features separately
test_emb = test_vec.iloc[:,:768]
test_icpc_enc = test_vec.iloc[:,768].to_numpy().reshape(-1,1)
test_mo_enc = test_vec.iloc[:,769:771]
test_age_scaled = test_vec.iloc[:,771].to_numpy().reshape(-1,1)

test_set = [test_emb, test_icpc_enc, test_mo_enc, test_age_scaled]

autoencoder = load_model(config.autoencoder)

train_reconstructions = autoencoder.predict(train_set)
mse_train = np.mean(np.power(train_vec - train_reconstructions,2),axis=1)
threshold = np.percentile(mse_train, 90)

test_reconstructions = autoencoder.predict(test_set)
mse_test = np.mean(np.power(test_vec - test_reconstructions,2), axis=1)
scaled_mse_test = (mse_test - np.mean(mse_train)) / np.std(mse_train)
anomalies = scaled_mse_test > threshold
anomaly_indices = np.where(anomalies)[0]

filtered_texts_df = test_df.iloc[anomaly_indices] # normal data
filtered_mse_test = mse_test[anomaly_indices]

filtered_texts_df_vec_emb = test_emb.iloc[anomaly_indices] # vectorized data
filtered_age = test_age_scaled[anomaly_indices]
filtered_texts_df_vec_emb.reset_index(drop=True, inplace=True)

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in custom_stopwords]
    return tokens



reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=config.random_seed, n_components=50)
reduced_data = reducer.fit_transform(filtered_texts_df_vec_emb)
combined_df_vec = np.concatenate([reduced_data, filtered_age], axis=1)

cosine_sim_matrix = cosine_similarity(combined_df_vec)
cosine_dist_matrix = (1 - cosine_sim_matrix).astype(np.float64)

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=5, metric='precomputed', approx_min_span_tree=True, cluster_selection_method='eom')
cluster_labels = clusterer.fit_predict(cosine_dist_matrix)
filtered_texts_df = filtered_texts_df.copy()
filtered_texts_df['cluster'] = cluster_labels
num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

cluster_documents = {}
for _,row in filtered_texts_df.iterrows():
    cluster_label = row['cluster']
    document = row['soep_text']
    processed_doc = preprocess(document)
    processed_doc_str = ' '.join(processed_doc)
    if cluster_label not in cluster_documents:
        cluster_documents[cluster_label] = []
    cluster_documents[cluster_label].append(processed_doc_str)

tfidf_vectorizer = TfidfVectorizer()
combined_cluster_docs = {label: ' '.join(docs) for label, docs in cluster_documents.items()}
combined_documents = list(combined_cluster_docs.values())
cluster_labels_unique = list(combined_cluster_docs.keys())
X_cluster = tfidf_vectorizer.fit_transform(combined_documents)
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = X_cluster.toarray()

cluster_tfidf_dict = {}
for idx, cluster_id in enumerate(cluster_labels_unique):
    word_tfidf = {feature_names[i]: tfidf_scores[idx, i] for i in range(len(feature_names))}
    sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)[:10]
    if config.english:
        translator = Translator()
        sorted_words = [(translator.translate(token[0]),token[1]) for token in sorted_words]
    cluster_tfidf_dict[cluster_id] = dict(sorted_words)


try:
    plt.scatter(reduced_data[:,0], reduced_data[:,1], c=cluster_labels)
    plt.show()
except:
    pass


print(f"{config.test_start_month}/{config.test_start_year} - {config.test_end_month}/{config.test_end_year}: Number of anomalies: {len(filtered_texts_df)} | Number of clusters: {num_clusters}")

geo_postcodes3 = pd.read_csv(config.postcode3)
geo_postcodes3['geometry'] = geo_postcodes3['geometry'].apply(wkt.loads)
geo_postcodes3_selected = geo_postcodes3[['postcode','geometry']]

silhouette_vals = silhouette_samples(combined_df_vec, cluster_labels, metric='cosine')

pdf_filename = f'results/clusters_{config.test_on}_{config.test_start_month}-{config.test_start_year}_{config.test_end_month}-{config.test_end_year}.pdf'
with PdfPages(pdf_filename) as pdf:
    for cluster in np.unique(cluster_labels):
        if cluster == -1:
            continue
        try:
            cluster_df = filtered_texts_df[filtered_texts_df['cluster'] == cluster]
            text_per_cluster = cluster_df['soep_text'].tolist()
            cluster_indices = (filtered_texts_df['cluster'] == cluster).values

            cluster_mse = filtered_mse_test[cluster_indices]
            cluster_mse = (cluster_mse - np.nanmin(mse_test)) / np.nanmax(mse_test) - np.nanmin(mse_test)
            mean_cluster_mse = np.nanmean(cluster_mse)

            cluster_sil_vals = silhouette_vals[cluster_indices]
            cluster_sil_score = np.mean(cluster_sil_vals)

            fig = plt.figure(figsize=(12,8))
            gs = gridspec.GridSpec(2,2,height_ratios=[1,1],width_ratios=[2,1])
            ax_left = plt.subplot(gs[:,0])
            ax_right_top = plt.subplot(gs[0,1])
            ax_right_bottom = plt.subplot(gs[1,1])

            fig.suptitle(f'{config.test_start_month}/{config.test_start_year} - {config.test_end_month}/{config.test_end_year} | Cluster: {cluster + 1}/{num_clusters} | Anomalies: {len(text_per_cluster)}\nMean normalized cluster MSE: {mean_cluster_mse:.4f} | Cluster silhouette score: {cluster_sil_score:.4f}')

            age_gender_count = filtered_texts_df[filtered_texts_df['cluster'] == cluster].groupby(['age_cat','sex']).size().unstack(fill_value=0)
            age_categories = ['0', '1-3', '4-5', '6-12', '13-17', '18-25', '26-40', '41-55', '56-70', '71+']
            age_gender_count = age_gender_count.reindex(age_categories, fill_value=0)
            bottom = np.zeros(len(age_gender_count))
            if config.english:
                gender_colors = {'M': '#0000FF','F':'#FFC0CB'}
            else:
                gender_colors = {'M': '#0000FF','V':'#FFC0CB'}
            for gender in age_gender_count.columns:
                ax_right_top.bar(age_gender_count.index,age_gender_count[gender],label=gender, color=gender_colors.get(gender, 'gray'),bottom=bottom,edgecolor='black')
                bottom += age_gender_count[gender].values

            ax_right_top.set_title('Demographics')
            ax_right_top.set_xlabel('Age')
            ax_right_top.set_ylabel('Frequency')
            ax_right_top.legend(title='Sex')
            ax_right_top.set_xticklabels(ax_right_top.get_xticklabels(), rotation=30, ha='right')

            postalcode_counts = cluster_df.groupby('postal_code').size().reset_index(name='count')
            merged_df = pd.merge(postalcode_counts, geo_postcodes3_selected, how='outer', left_on='postal_code',
                                         right_on='postcode')
            merged_df['count'] = merged_df['count'].fillna(0)
            geo_df = gpd.GeoDataFrame(merged_df, geometry='geometry')
            pd.set_option('display.max_columns', None)
            geo_df[geo_df['count'] > 0].plot(column='count', cmap='OrRd', legend=True, ax=ax_right_bottom)
            geo_df[geo_df['count'] == 0].plot(color='darkgray', ax=ax_right_bottom)
            ax_right_bottom.set_axis_off()
            ax_right_bottom.set_title('Postal code')

            #print(cluster_tfidf_dict[cluster])

            cluster_tfidf = cluster_tfidf_dict[cluster]
            words = list(cluster_tfidf.keys())
            scores = list(cluster_tfidf.values())

            ax_left.barh(words,scores,color='orange', edgecolor='black')

            ax_left.set_xlabel('TF-IDF score')
            ax_left.set_ylabel('Words')
            ax_left.set_title('Topic')
            ax_left.invert_yaxis()
            for i, score in enumerate(scores):
                ax_left.text(score * 0.5, i, f'{score:.3f}', va='center', fontsize=10)


            # wordcloud = WordCloud(background_color='white').generate_from_frequencies(cluster_tfidf_dict[cluster])
            # axs[1,0].imshow(wordcloud)


            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            #print('_' * 40)
        except ValueError:
            print('No clusters found.')
print(f'Clusters saved successfully in {pdf_filename}.')
