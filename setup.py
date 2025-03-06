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

import os
from transformers import BertTokenizer, BertModel, MarianMTModel, MarianTokenizer
import nltk

# Define model directories
MODEL_DIR = "models/"
BERTJE_PATH = os.path.join(MODEL_DIR, "bertje/")
OPUS_NL_EN_PATH = os.path.join(MODEL_DIR, "opus_nl_en/")

# Ensure the directories exist
os.makedirs(BERTJE_PATH, exist_ok=True)
os.makedirs(OPUS_NL_EN_PATH, exist_ok=True)

# Download and save BERTje (Overwrites existing files)
print("Downloading BERTje model...")
bertje_tokenizer = BertTokenizer.from_pretrained("GroNLP/bert-base-dutch-cased")
bertje_model = BertModel.from_pretrained("GroNLP/bert-base-dutch-cased")
bertje_tokenizer.save_pretrained(BERTJE_PATH)
bertje_model.save_pretrained(BERTJE_PATH)
print(f"BERTje model saved to: {BERTJE_PATH}")

# Download and save OPUS MT Dutch-to-English (Overwrites existing files)
print("Downloading OPUS MT Dutch-to-English model...")
opus_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
opus_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-nl-en")
opus_tokenizer.save_pretrained(OPUS_NL_EN_PATH)
opus_model.save_pretrained(OPUS_NL_EN_PATH)
print(f"OPUS MT model saved to: {OPUS_NL_EN_PATH}")

# Ensure NLTK is installed but do not download stopwords
print("Installing NLTK tokenizer...")
nltk.download('punkt')
print("NLTK installed. Custom stopwords file will be used instead.")

print("Setup complete. Models are ready to use.")
