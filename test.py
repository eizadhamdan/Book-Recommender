import pandas as pd
books = pd.read_csv("C:\\Codes\\GenAI\\Book-Recommender\\data\\books_with_emotions.csv")
print("Required columns check:")
required_cols = ['isbn13', 'title', 'authors', 'description', 'simple_categories']
for col in required_cols:
    print(f"  {col}: {'✓' if col in books.columns else '✗'}")
print(f"\nAll columns: {books.columns.tolist()}")