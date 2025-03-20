# Semantic Book Recommender

This project is a content-based book recommendation system powered by Hugging Face embeddings, LangChain, and Qdrant. It transforms book descriptions into vector embeddings and retrieves similar books using semantic search. The project is deployed on Hugging Face Spaces.

Live Demo: [App Demo](https://huggingface.co/spaces/hemesh0204/book_recommender)

## Features

- Semantic search using Hugging Face embeddings  
  - Uses `sentence-transformers/all-MiniLM-L6-v2` to embed book descriptions.  
- Vector database with Qdrant  
  - Stores and retrieves books efficiently using cosine similarity.  
- Gradio-powered user interface  
  - Interactive interface for finding book recommendations.  
- Emotion and category filtering  
  - Allows filtering based on book categories and emotions such as Happy, Surprising, Angry, etc.  
- Deployed on Hugging Face Spaces  
  - Publicly accessible without additional setup.  

## How It Works

1. Preprocessing the book dataset  
   - Loads book data from `books_with_emotions.csv`  
   - Cleans descriptions and metadata  

2. Creating vector embeddings  
   - Converts book descriptions into embeddings using `sentence-transformers/all-MiniLM-L6-v2`  

3. Storing in Qdrant vector database  
   - Stores book embeddings and metadata including ISBN, title, and author  

4. Retrieving similar books  
   - Uses cosine similarity search to find books matching the user's query  

5. Gradio user interface for recommendations  
   - Users input a description, select a category or tone, and receive relevant book suggestions  

