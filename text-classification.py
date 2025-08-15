from openai import OpenAI
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


books = pd.read_csv("books_cleaned.csv")
 
books["categories"].value_counts().reset_index()
 
books["categories"].value_counts().reset_index().query("count >= 50")
 
books[books["categories"] == "Juvenile Fiction"]

 
catgeory_mapping = {
    "Fiction": "Fiction",
    "Juvenile Fiction": "Children's Fiction",
    "Biography & Autobiography": "Nonfiction",
    "History": "Nonfiction",
    "Literacy Criticism": "Nonfiction",
    "Philosophy": "Nonfiction",
    "Religion": "Nonfiction",
    "Comics & Graphic Novels": "Fiction",
    "Drama": "Fiction",
    "Juvenile Nonfiction": "Children's Nonfiction",
    "Science": "Nonfiction",
    "Poetry": "Nonfiction",
    }

books["simple_categories"] = books["categories"].map(catgeory_mapping)

 
books[~(books["simple_categories"].isna())]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

 
def gpt_zero_shot_classify(sequence, candidate_labels):
    prompt = f"""
    You are a helpful assistant. Classify the following text into one of these categories: {candidate_labels}.

    Text: "{sequence}"

    Respond only with the label from the list that best fits the text.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


fiction_categories = ["Fiction", "Nonfiction"]
# pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[0]

# pipe(sequence, fiction_categories)

predicted_label = gpt_zero_shot_classify(sequence, fiction_categories)
print(predicted_label)
 
# max_index = np.argmax(pipe(sequence, fiction_categories)["scores"])
# max_label = pipe(sequence, fiction_categories)["labels"][max_index]

 
# print(max_label)

 
def generate_predictions(sequence, categories):
    return gpt_zero_shot_classify(sequence, categories)

 
# def generate_predictions(sequence, categories):
#     predictions = pipe(sequence, categories)
#     max_index = np.argmax(predictions["scores"])
#     max_label = predictions["labels"][max_index]

#     return max_label


actual_categories = []
predicted_categories = []

# Classify first 300 Fiction descriptions
for i in tqdm(range(0, 300)):
    sequence = books.loc[books["simple_categories"] == "Fiction", "description"].reset_index(drop=True)[i]
    predicted_categories.append(generate_predictions(sequence, fiction_categories))
    actual_categories.append("Fiction")

 
# Classify first 300 Nonfiction descriptions
for i in tqdm(range(0, 300)):
    sequence = books.loc[books["simple_categories"] == "Nonfiction", "description"].reset_index(drop=True)[i]
    predicted_categories.append(generate_predictions(sequence, fiction_categories))
    actual_categories.append("Nonfiction")

 
predictions_df = pd.DataFrame({"actual_categories": actual_categories, "predicted_categories": predicted_categories})

 
print(predictions_df.head())

 
predictions_df["correct_prediction"] = predictions_df["actual_categories"] == predictions_df["predicted_categories"]


 
accuracy = predictions_df["correct_prediction"].sum() / len(predictions_df)
print("Accuracy:", accuracy)


# # Classify missing categories
 
isbns = []
predicted_categories = []

missing_categories = books.loc[books["simple_categories"].isna(), ["isbn13", "description"]].reset_index(drop=True)

for i in tqdm(range(len(missing_categories))):
    sequence = missing_categories["description"][i]
    predicted_categories.append(generate_predictions(sequence, fiction_categories))
    isbns.append(missing_categories["isbn13"][i])
 
missing_predicted_df = pd.DataFrame({"isbn13": isbns, "predicted_categories": predicted_categories})

 
print(missing_predicted_df)


books = pd.merge(books, missing_predicted_df, on="isbn13", how="left")
books["simple_categories"] = np.where(
    books["simple_categories"].isna(), books["predicted_categories"], books["simple_categories"]
)
books = books.drop(columns=["predicted_categories"])
 
books[books["categories"].str.lower().isin([
    "romance",
    "science fiction",
    "scifi",
    "fantasy",
    "horror",
    "mystery",
    "thriller",
    "comedy",
    "crime",
    "historical"
])]
 
books.to_csv("books_with_categories.csv", index=False)
