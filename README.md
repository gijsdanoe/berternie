# berternie

Repository for the BERT and ERNIE framework for the anomaly detection and clustering of infectious disease-related GP texts.

## Project structure
- `data/` → Raw, preprocessed, artificial and external data (stopwords and postal codes).
- `models/` → Autoencoder model, BERTje, OPUS-MT and NLTK.
- `results/` → Clustering results.
- `src/` → All source code.

# Setup

## Install dependencies

```pip install -r requirements.txt```

## Run setup.py

```python setup.py```

This will install all required models—BERTje, OPUS MT (Dutch-to-English), and NLTK inside the `models/` directory.

## Configuration

All configurations and paths are set in the config.py file. If needed, modify. Then run:

```python3 config.py```

Here, you can select if you want the output to be Dutch or English, if you want to include artificial data, and which periods you want to train and test the model on.

# Usage

1. To preprocess the raw data files, first run:

```python3 preprocessing.py```

2. Then run:

```python3 train_test.py```

This will create the train and test files, based on the periods set in `config.py`.

3. Train the autoencoder:

```python3 train_autoencoder.py```

This will create a file `autoencoder_model.h5` in the `models/` directory.

4. Finally, run:

```python3 clustering.py```

This will run the autoencoder on the test set to flag anomalies, and run the clustering model, which then outputs the cluster plots in the `results/` directory.



