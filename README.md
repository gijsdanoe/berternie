# berternie

Repository for the BERT and ERNIE framework for the anomaly detection and clustering of infectious disease-related GP texts.

## Project structure
- `data/` → Raw, preprocessed, train and test, artificial and external data (stopwords and postal codes).
- `models/` → Autoencoder model, BERTje, OPUS-MT and NLTK.
- `results/` → Clustering results.
- `src/` → All source code.

Due to privacy regulations, the real datasets cannot be shared.

## License

This project is licensed under the **GNU General Public License v2 (GPL v2)**.  
You are free to use, modify, and distribute this software under the terms of the **GPL-2.0** license.

See the [LICENSE](./LICENSE) file for full details.

### Summary of GPL v2:
- ✅ **You may** use and modify this project for personal and commercial purposes.
- ✅ **You may** distribute copies, but they must also be licensed under **GPL v2**.
- ❌ **You may not** use this software for patent-related claims.
- ❌ **There is no warranty**—use at your own risk.

For more details, visit the **[GNU GPL v2 License](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html)** page.

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



