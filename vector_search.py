from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import pandas as pd


load_dotenv()

books = pd.read_csv("books_cleaned.csv")
print(books)

books["tagged_description"].to_csv("tagged_description.txt",
                                   sep="\n",
                                   index=False,
                                   header=False)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()
text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
documents = text_splitter.split_documents(raw_documents)

