from train import train


def main():

    model = train(
        corpus_path="data/corpus.txt",
        model_dir="models",
        vector_size=150,
        window=5,
        min_count=3,
        sg=1,
        epochs=20
    )
    print("Vocabulary size:", len(model.wv))


if __name__ == "__main__":
    main()
