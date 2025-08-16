import pandas as pd
from transformers import pipeline
import numpy as np
from tqdm import tqdm


books = pd.read_csv("books_with_categories.csv")


classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None, device=0)


print(classifier("I love this!"))


print(books["description"][0])


model_name = "j-hartmann/emotion-english-distilroberta-base"
local_dir = "./emotion_model"

classifier.model.save_pretrained(local_dir)
classifier.tokenizer.save_pretrained(local_dir)

print("Model saved locally at", local_dir)


print(classifier(books["description"][0]))


print(classifier(books["description"][0].split(".")))


sentences = books["description"][0].split(".")
predictions = classifier(sentences)


print(sentences[0])
print(predictions[0])


print(sentences[3])
print(predictions[3])


print(sorted(predictions[0], key=lambda x: x["score"]))


print(sorted(predictions[0], key=lambda x: x["label"]))


emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

isbn = []

emotion_scores = {label: [] for label in emotion_labels}


def calculate_max_emotion_scores(predictions):
    per_emotion_scores = {label: [] for label in emotion_labels}
    for prediction in predictions:
        sorted_predictions = sorted(prediction, key=lambda x: x["score"])
        for index, label in enumerate(emotion_labels):
                per_emotion_scores[label].append(sorted_predictions[index]["score"])
    return {label: np.max(scores) for label, scores in per_emotion_scores.items()}


for i in range(10):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)

    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])


print(emotion_scores)


emotion_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
isbn = []
emotion_scores = {label: [] for label in emotion_labels}

for i in tqdm(range(len(books))):
    isbn.append(books["isbn13"][i])
    sentences = books["description"][i].split(".")
    predictions = classifier(sentences)
    max_scores = calculate_max_emotion_scores(predictions)

    for label in emotion_labels:
        emotion_scores[label].append(max_scores[label])


emotions_df = pd.DataFrame(emotion_scores)
emotions_df["isbn13"] = isbn


print(emotions_df.head())


books = pd.merge(books, emotions_df, on="isbn13")


print(books)


books.to_csv("books_with_emotions.csv", index=False)
