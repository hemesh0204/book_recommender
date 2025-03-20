import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import gradio as gr

from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Load books dataset
books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Initialize Qdrant Client
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME")  

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


hf = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant = Qdrant(
    client=client,
    collection_name=QDRANT_COLLECTION_NAME,
    embeddings=hf  
)



def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    
    # 🔹 Generate Query Embedding
    query_embedding = hf.embed_query(query)

    # 🔹 Retrieve top `initial_top_k` similar books from Qdrant
    recs = qdrant.similarity_search_by_vector(query_embedding, k=initial_top_k)

    

    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)


    # 🔹 Apply category filtering
    if category and category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
        print(f"Filtered {len(book_recs)} books after category filter")  # Debugging
    else:
        book_recs = book_recs.head(final_top_k)

    # 🔹 Apply tone-based sorting
    tone_mapping = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_mapping:
        book_recs = book_recs.sort_values(by=tone_mapping[tone], ascending=False)


    return book_recs.head(final_top_k)


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    
    print(f"Final Recommendations: {len(recommendations)} books")  # Debugging

    if recommendations.empty:
        print("No books found after filtering.")
        return [("cover-not-found.jpg", "No books found matching your query.")]

    results = []
    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    print(f"Returning {len(results)} books to Gradio")  # Debugging
    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Book recommender")

    with gr.Row():
        user_query = gr.Textbox(label = "Please enter a description of a book:",
                                placeholder = "e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices = categories, label = "Select a category:", value = "All")
        tone_dropdown = gr.Dropdown(choices = tones, label = "Select an emotional tone:", value = "All")
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label = "Recommended books", columns = 8, rows = 2)

    submit_button.click(fn = recommend_books,
                        inputs = [user_query, category_dropdown, tone_dropdown],
                        outputs = output)


if __name__ == "__main__":
    dashboard.launch()
