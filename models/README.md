
## Darija Word Embedding and Generative Model

This project develops and demonstrates an Arabic Word2Vec model for finding similar words and a PyTorch-based generative model (LSTM) for sentence completion using a Darija corpus.

### Project Structure

-   `/content/Darija-Word-Embedding/data/corpus.txt`: The text corpus used for training.
-   `/content/Darija-Word-Embedding/models/`: Directory where trained models are saved.
-   `preprocess.py`: Contains functions for Arabic text normalization, tokenization, and corpus loading.
-   This notebook: Contains all the steps for data preparation, model training, and demonstration.

### Setup

No specific installation steps are needed beyond what's handled within the notebook, as `gensim`, `torch`, `tqdm`, and `pyarabic` are either pre-installed or installed dynamically.

### Usage

The notebook walks through the following steps:

1.  **Project Directory Cleanup**: Ensures a clean working environment by removing old files, preserving only the `data` folder.

2.  **Text Preprocessing**:
    *   `preprocess.py` is dynamically created/loaded with functions like `normalize_arabic`, `tokenize`, and `load_corpus`.
    *   The `load_corpus` function reads `corpus.txt`, performs normalization and tokenization, and returns a list of preprocessed sentences.
    *   A vocabulary is built from this preprocessed corpus, mapping words to numerical indices for the generative model.

3.  **Word2Vec Model Training**:
    *   A `gensim.models.Word2Vec` model is trained on the `preprocessed_corpus`.
    *   **Key Parameters**: `vector_size` (embedding dimension), `window` (context window size), `min_count` (minimum word frequency), `sg` (CBOW or Skip-gram), `epochs`.
    *   The trained model (`word2vec_model.bin`) and its word vectors (`arabic_vectors.kv`) are saved to the `models/` directory.

4.  **Similar Word Prediction**:
    *   The trained Word2Vec model is loaded.
    *   The `model.wv.most_similar(word, topn=N)` method is used to find and display the top N words semantically similar to a given `test_word` (e.g., "الاسلام").

5.  **PyTorch Generative Model (LSTM) Training**:
    *   **Data Preparation**: The numerical sequences are prepared into input-target pairs (`X`, `y`) with a defined `sequence_length` and batched using `DataLoader`.
    *   **Model Architecture**: A `GenerativeLSTM` class is defined using `torch.nn.Module`, featuring an `Embedding` layer, `LSTM` layers, and a `Linear` output layer. It includes dropout for regularization.
    *   **Training Loop**: The model is trained using `CrossEntropyLoss` and `Adam` optimizer over several epochs. Hidden states are initialized dynamically for each batch.

6.  **Sentence Completion**:
    *   A `generate_text` function uses the trained `GenerativeLSTM` model to predict the next word in a sequence.
    *   It takes a `start_sequence`, `max_len` (maximum words to generate), and `temperature` (for sampling diversity).
    *   Crucially, it suppresses the prediction of the padding token to ensure meaningful word generation.

### Key Methods and Features

**Preprocessing (`preprocess.py` functions)**:
-   `strip_diacritics(text)`: Removes Arabic diacritics.
-   `normalize_arabic(text)`: Applies a full normalization pipeline (diacritics, Tatweel, Alef/Teh Marbuta/Yeh variants, non-Arabic chars, whitespace).
-   `tokenize(text, min_token_len=2)`: Normalizes and splits text into tokens.
-   `load_corpus(filepath, min_sentence_len=3)`: Loads a corpus from a file, preprocesses, and tokenizes sentences.

**Word2Vec Model (`gensim.models.Word2Vec` or `Word2VecTrainer` class)**:
-   `Word2VecTrainer.__init__(...)`: Initializes Word2Vec model parameters (vector size, window, min count, etc.).
-   `Word2VecTrainer.train(corpus)`: Trains the Word2Vec model on the provided corpus.
-   `Word2VecTrainer.save_model(model_name)`: Saves the trained model.
-   `model.wv.most_similar(positive=[], negative=[], topn=10)`: Finds words most similar to the given words/vectors.

**PyTorch Generative Model (`GenerativeLSTM` class)**:
-   `GenerativeLSTM.__init__(vocab_size, embedding_dim, hidden_dim, num_layers)`: Defines the LSTM architecture.
-   `GenerativeLSTM.forward(x, hidden)`: Defines the forward pass of the network.
-   `GenerativeLSTM.init_hidden(batch_size, device)`: Initializes the hidden and cell states for the LSTM.
-   `generate_text(model, start_sequence, max_len=50, temperature=1.0)`: Function to generate text given a starting sequence.

### How to Feed New Data (without losing previous learning)

**For Word2Vec (continuous training)**:
`gensim` models can be updated incrementally. To train on new data without losing previous learning:

1.  Load your existing `word2vec_model`.
2.  Prepare your `new_data_corpus` using the same `preprocess.py` functions.
3.  **Update Vocabulary (if new words appear)**:
    ```python
    word2vec_model.build_vocab(new_data_corpus, update=True)
    ```
4.  **Continue Training**: Then, call the `train` method again with the new corpus:
    ```python
    word2vec_model.train(new_data_corpus, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.epochs)
    ```
    *Note*: `total_examples` and `epochs` should be set appropriately for the combined (original + new) data or just the new data, depending on your goal.

**For PyTorch Generative Model**:
To incorporate new data into the generative model, you generally have two main approaches:

1.  **Retrain on Combined Data**: The most straightforward way is to combine your `preprocessed_corpus` with the `new_data_corpus`, recreate the `encoded_corpus`, `TensorDataset`, and `DataLoader`, and then retrain the existing `generative_model` from its current state on this larger, combined dataset. This allows the model to learn from both old and new data.

    ```python
    # Example: Combine old and new preprocessed sentences
    combined_corpus = preprocessed_corpus + new_data_corpus

    # Rebuild vocabulary and re-encode combined corpus
    # (Ensure consistent word_to_idx, idx_to_word mappings)
    # ... (steps for vocabulary building and encoding as in the notebook)

    # Re-create TensorDataset and DataLoader for combined_corpus
    # ...

    # Continue training the existing generative_model on the new dataloader
    # generative_model.train()
    # for epoch in range(additional_epochs):
    #    ...
    ```

2.  **Fine-tuning**: If the new data is significantly different or you want to rapidly adapt the model, you can fine-tune the already trained model on just the `new_data_corpus` for a few epochs. This might require careful adjustment of the learning rate.

### Insights & Next Steps

-   The Word2Vec model provides valuable word embeddings for semantic similarity tasks.
-   The PyTorch generative model demonstrates basic sentence completion. For more sophisticated generation (e.g., diverse and coherent long sequences), consider exploring advanced architectures like Transformers, beam search decoding, or techniques like reinforcement learning for text generation.
