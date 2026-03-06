from .train import _load_model, _cosine_sim, predict_next_word


def test_vocab():
    # Check how many words the model learned.
    model, w2i, _ = _load_model()
    print(f"Vocab size: {len(w2i)}")


def test_similar(word, top_k=5):
    # Find the most similar words to a given word.
    model, w2i, i2w = _load_model()

    if word not in w2i:
        print(f"'{word}' not in vocabulary.")
        return

    vec    = model.get_vector(w2i[word])
    scores = [
        (i2w[i], _cosine_sim(vec, model.get_vector(i)))
        for i in i2w if i2w[i] != word
    ]
    scores.sort(key=lambda x: x[1], reverse=True)

    print(f"Top {top_k} words similar to '{word}':")
    for w, score in scores[:top_k]:
        print(f"  {w:<20} {score:.4f}")


def test_predict(sentence, length=5):
    # Predict the next words for a sentence and print results.
    print(f"Predictions for: '{sentence}'")
    results = predict_next_word(sentence, length=length)
    for word, score in results:
        print(f"  {word:<20} {score:.4f}")
