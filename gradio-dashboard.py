import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import gradio as gr
import os
import sys

load_dotenv()

# Load books data
print("Loading books data...")
try:
    books = pd.read_csv("data/books_with_emotions.csv")
    print(f"Loaded {len(books)} books")
except Exception as e:
    print(f"Error loading books data: {e}")
    sys.exit(1)

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Initialize local embedding model
def get_embedding_model():
    """Initialize a lightweight local embedding model"""
    print("Initializing local embedding model...")
    try:
        # Using all-MiniLM-L6-v2 - lightweight, fast, and good performance
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
            encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity search
        )
        
        print(f"Successfully loaded model: {model_name}")
        return embeddings
        
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        print("Trying fallback model...")
        try:
            # Fallback to even smaller model
            model_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print(f"Successfully loaded fallback model: {model_name}")
            return embeddings
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")
            return None

# Initialize database with error handling
def initialize_database():
    """Initialize or load the Chroma database"""
    db_path = "db_books_local"  # Changed path to avoid conflicts
    
    # Get embedding model
    embeddings = get_embedding_model()
    if embeddings is None:
        print("Failed to load embedding model")
        return None
    
    try:
        # Check if database already exists
        if os.path.exists(db_path) and os.listdir(db_path):
            print("Loading existing database...")
            db_books = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            print("Database loaded successfully!")
            return db_books
        else:
            print("Creating new database...")
            print("Loading documents...")
            raw_documents = TextLoader("data/tagged_description.txt", encoding="utf-8").load()
            print(f"Loaded {len(raw_documents)} raw documents")
            
            print("Splitting documents...")
            text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
            documents = text_splitter.split_documents(raw_documents)
            print(f"Created {len(documents)} chunks")
            
            print("Creating embeddings and database... This may take a while...")
            print("Note: First run will download the model (~90MB) and create embeddings")
            db_books = Chroma.from_documents(
                documents,
                embeddings,
                persist_directory=db_path
            )
            print("Database created and persisted successfully!")
            return db_books
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        return None

# Initialize database
print("Initializing vector database...")
db_books = initialize_database()

if db_books is None:
    print("Failed to initialize database. Exiting...")
    sys.exit(1)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    try:
        print(f"Searching for: '{query}' with category: {category}, tone: {tone}")
        print("About to call db_books.similarity_search...")
        
        # Try with a smaller k first to test if the database is working
        try:
            print("Testing with k=5...")
            test_recs = db_books.similarity_search(query, k=5)
            print(f"Test search successful - found {len(test_recs)} results")
            
            # Now try the full search
            print(f"Running full search with k={initial_top_k}...")
            recs = db_books.similarity_search(query, k=initial_top_k)
            print(f"Found {len(recs)} similarity matches")
            
        except Exception as e:
            print(f"Vector search failed: {e}")
            print("Falling back to simple text search...")
            return fallback_text_search(query, category, tone, final_top_k)
        
        books_list = []
        for i, rec in enumerate(recs):
            try:
                if i < 5:  # Only print first 5 for debugging
                    print(f"Processing result {i+1}: {rec.page_content[:100]}...")
                
                # More robust parsing of the page content
                content = rec.page_content.strip()
                
                # Handle different formats:
                # Format 1: 9780688085872 Description...
                # Format 2: "9780688085872 Description..."
                if content.startswith('"'):
                    # Remove the leading quote and split
                    content = content[1:]  # Remove leading quote
                    if ' ' in content:
                        book_id_str = content.split(' ')[0]
                    else:
                        book_id_str = content.split('"')[0] if '"' in content else content
                else:
                    # No leading quote
                    if ' ' in content:
                        book_id_str = content.split(' ')[0]
                    else:
                        book_id_str = content
                
                # Clean and convert to int
                book_id_str = book_id_str.strip().replace('"', '')
                if book_id_str and book_id_str.isdigit():
                    book_id = int(book_id_str)
                    books_list.append(book_id)
                    if i < 5:
                        print(f"Extracted book ID: {book_id}")
                else:
                    if i < 10:  # Show more parsing errors for debugging
                        print(f"Could not parse book ID from: '{book_id_str}' (original: {content[:50]}...)")
                        
            except (ValueError, IndexError) as e:
                print(f"Error parsing book ID from content: {content[:50]}... Error: {e}")
                continue
        
        print(f"Extracted {len(books_list)} valid book IDs")
        
        if not books_list:
            print("No valid book IDs found - falling back to text search")
            return fallback_text_search(query, category, tone, final_top_k)
        
        book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)
        print(f"Found {len(book_recs)} matching books in database")

        if category != "All" and not book_recs.empty:
            book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
            print(f"After category filter: {len(book_recs)} books")
        
        if tone != "All" and not book_recs.empty:
            if tone == "Happy" and "joy" in book_recs.columns:
                book_recs = book_recs.sort_values(by="joy", ascending=False)
            elif tone == "Surprising" and "surprise" in book_recs.columns:
                book_recs = book_recs.sort_values(by="surprise", ascending=False)
            elif tone == "Angry" and "anger" in book_recs.columns:
                book_recs = book_recs.sort_values(by="anger", ascending=False)
            elif tone == "Suspenseful" and "fear" in book_recs.columns:
                book_recs = book_recs.sort_values(by="fear", ascending=False)
            elif tone == "Sad" and "sadness" in book_recs.columns:
                book_recs = book_recs.sort_values(by="sadness", ascending=False)
            print(f"After tone filter: {len(book_recs)} books")

        return book_recs
        
    except Exception as e:
        print(f"Error in retrieve_semantic_recommendations: {e}")
        import traceback
        traceback.print_exc()
        print("Falling back to simple text search...")
        return fallback_text_search(query, category, tone, final_top_k)

def fallback_text_search(query, category, tone, final_top_k):
    """Fallback search using simple text matching"""
    try:
        print("Using fallback text search...")
        query_lower = query.lower()
        
        # Search in title and description
        title_mask = books["title"].str.lower().str.contains(query_lower, na=False)
        desc_mask = books["description"].str.lower().str.contains(query_lower, na=False)
        
        # Combine masks
        mask = title_mask | desc_mask
        
        book_recs = books[mask].head(final_top_k)
        print(f"Text search found {len(book_recs)} books matching '{query}'")
        
        if len(book_recs) == 0:
            # Try broader search with individual words
            words = query_lower.split()
            if len(words) > 1:
                print(f"No exact matches found, trying individual words: {words}")
                word_masks = []
                for word in words:
                    if len(word) > 2:  # Skip very short words
                        word_mask = (
                            books["title"].str.lower().str.contains(word, na=False) |
                            books["description"].str.lower().str.contains(word, na=False)
                        )
                        word_masks.append(word_mask)
                
                if word_masks:
                    # Combine with OR logic
                    combined_mask = word_masks[0]
                    for mask in word_masks[1:]:
                        combined_mask = combined_mask | mask
                    
                    book_recs = books[combined_mask].head(final_top_k)
                    print(f"Word-based search found {len(book_recs)} books")
        
        if category != "All" and not book_recs.empty:
            book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
            print(f"After category filter: {len(book_recs)} books")
        
        if tone != "All" and not book_recs.empty:
            if tone == "Happy" and "joy" in book_recs.columns:
                book_recs = book_recs.sort_values(by="joy", ascending=False)
            elif tone == "Surprising" and "surprise" in book_recs.columns:
                book_recs = book_recs.sort_values(by="surprise", ascending=False)
            elif tone == "Angry" and "anger" in book_recs.columns:
                book_recs = book_recs.sort_values(by="anger", ascending=False)
            elif tone == "Suspenseful" and "fear" in book_recs.columns:
                book_recs = book_recs.sort_values(by="fear", ascending=False)
            elif tone == "Sad" and "sadness" in book_recs.columns:
                book_recs = book_recs.sort_values(by="sadness", ascending=False)
            print(f"After tone filter: {len(book_recs)} books")
        
        print(f"Final fallback search result: {len(book_recs)} books")
        return book_recs
        
    except Exception as e:
        print(f"Error in fallback search: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def recommend_books(
        query: str,
        category: str,
        tone: str,
):
    try:
        print(f"recommend_books called with: query='{query}', category='{category}', tone='{tone}'")
        
        if not query.strip():
            print("Empty query provided")
            return []
        
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        
        if recommendations.empty:
            print("No recommendations found")
            return []
        
        results = []
        print(f"Processing {len(recommendations)} recommendations...")

        for idx, (_, row) in enumerate(recommendations.iterrows()):
            try:
                description = row["description"] if pd.notna(row["description"]) else "No description available"
                truncated_desc_split = description.split()
                truncated_description = " ".join(truncated_desc_split[:30]) + "..." if len(truncated_desc_split) > 30 else description

                authors = row["authors"] if pd.notna(row["authors"]) else "Unknown author"
                authors_split = authors.split(";")
                if len(authors_split) == 2:
                    authors_str = f"{authors_split[0]}, {authors_split[1]}"
                elif len(authors_split) > 2:
                    authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
                else:
                    authors_str = authors

                title = row["title"] if pd.notna(row["title"]) else "Unknown title"
                caption = f"{title} by {authors_str}: {truncated_description}"
                
                # Handle thumbnail
                thumbnail = row["large_thumbnail"] if pd.notna(row["large_thumbnail"]) else "cover-not-found.jpg"
                
                results.append((thumbnail, caption))
                print(f"Processed book {idx + 1}: {title}")
                
            except Exception as e:
                print(f"Error processing book {idx}: {e}")
                continue
        
        print(f"Successfully processed {len(results)} book recommendations")
        return results
        
    except Exception as e:
        print(f"Error in recommend_books: {e}")
        import traceback
        traceback.print_exc()
        return []

# Prepare dropdown options
try:
    categories = ["All"] + sorted([cat for cat in books["simple_categories"].unique() if pd.notna(cat)])
    tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
except Exception as e:
    print(f"Error preparing dropdown options: {e}")
    categories = ["All"]
    tones = ["All"]

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# My Book Recommender")
    gr.Markdown("*Powered by local Hugging Face embeddings*")

    with gr.Row():
        user_query = gr.Textbox(
            label="What book are you looking for?", 
            placeholder="Enter your query here...",
            scale=2
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Select a category:", 
            value="All",
            scale=1
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Select an emotional tone:", 
            value="All",
            scale=1
        )  
        submit_button = gr.Button("Get Recommendations", scale=1)

    gr.Markdown("### Recommendations")
    recommendations_output = gr.Gallery(
        label="Recommended Books", 
        columns=4, 
        rows=4,
        height="auto"
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=recommendations_output
    )
    
    # Allow Enter key to trigger search
    user_query.submit(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=recommendations_output
    )

if __name__ == "__main__":
    print("Starting Gradio dashboard...")
    try:
        dashboard.launch(
            server_name="localhost", 
            server_port=7860, 
            share=True,
            show_error=True
        )
    except Exception as e:
        print(f"Error launching dashboard: {e}")
