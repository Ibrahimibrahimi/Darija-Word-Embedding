from model import train, embed_dataset, predict_next_word

# Step 1: Train the model on the dataset.
train()

# Step 2: Save all sentence vectors to data/embedding.csv.
embed_dataset()

# Step 3: Predict next words for a sentence.
results = predict_next_word("ana machi", length=5)

for word, score in results:
    print(f"{word:<20} {score:.4f}")
