from flask import Flask, render_template, request, jsonify
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import os
import torch

app = Flask(__name__)

# Configuration
EMBEDDINGS_FILE = "book_embeddings.npy"
DATA_FILE = "books.csv"
REQUIRED_COLUMNS = ["title", "author", "description", "genres", "rating"]

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)


def prepare_data():
    """Load and prepare data, generating embeddings if needed"""
    # Load data
    df = pd.read_csv(DATA_FILE)

    # Check required columns
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Clean data
    df["combined_text"] = (df["title"].fillna("") + " " +
                           df["author"].fillna("") + " " +
                           df["description"].fillna(""))
    df.dropna(subset=["combined_text", "rating"], inplace=True)

    # Generate or load embeddings
    if os.path.exists(EMBEDDINGS_FILE):
        book_embeddings = np.load(EMBEDDINGS_FILE)
        if not isinstance(book_embeddings, torch.Tensor):
            book_embeddings = torch.from_numpy(book_embeddings).to(device)
    else:
        print("Generating new embeddings...")
        book_embeddings = sbert_model.encode(
            df["combined_text"].tolist(),
            convert_to_tensor=True,
            device=device
        )
        np.save(EMBEDDINGS_FILE, book_embeddings.cpu().numpy())

    book_embeddings = util.normalize_embeddings(book_embeddings)
    return df, book_embeddings


# Load data and embeddings
df, book_embeddings = prepare_data()

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your api key")
GROQ_MODEL = "llama3-70b-8192"
headers = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}

# System message for the book recommender
system_message = {
    "role": "system",
    "content": """You're a friendly book recommender. Your goal is to understand the user's reading preferences through conversation.

    Conversation Rules:
    1. Start by asking what kind of books they generally enjoy
    2. Ask at least 3-4 follow-up questions about:
       - Favorite genres/authors
       - Recent books they liked/disliked
       - Reading mood/purpose (entertainment, learning, etc.)
       - Any deal-breakers (length, themes, etc.)
    3. Only offer recommendations when you have a clear understanding of their preferences
    4. When recommending, say: "Based on our conversation, I recommend..."
    """
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    conversation = request.json.get("conversation", [
        {"role": "system",
         "content": """You're a book recommendation assistant. Follow this protocol:
         1. Ask at least 4-5 questions before recommending
         2. Only recommend when saying "Based on our conversation..."
         3. Never recommend and ask a question in the same response"""}
    ])

    conversation.append({"role": "user", "content": user_message})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": GROQ_MODEL,
            "messages": conversation,
            "temperature": 0.7,
            "max_tokens": 150
        }
    ).json()

    assistant_reply = response["choices"][0]["message"]["content"]
    conversation.append({"role": "assistant", "content": assistant_reply})

    # More strict recommendation detection
    should_recommend = (
        "based on our conversation" in assistant_reply.lower() and
        len([msg for msg in conversation if msg["role"] == "user"]) >= 4
    )

    return jsonify({
        "reply": assistant_reply,
        "conversation": conversation,
        "should_recommend": should_recommend
    })

@app.route("/recommend", methods=["POST"])
def recommend():
    conversation = request.json.get("conversation")

    # Extract preferences
    prompt = """
    Summarize the user's book preferences in 1-2 sentences, focusing on:
    - Genres (e.g., sci-fi, romance)
    - Themes (e.g., dystopian, adventure)
    - Writing style (e.g., fast-paced, poetic)
    - Preferred rating level (e.g., highly-rated books)
    Return ONLY the summary. Example:
    "The user likes highly-rated psychological thrillers with unpredictable plots."
    """

    messages = conversation + [{"role": "user", "content": prompt}]
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 100
        }
    ).json()

    preference_text = response["choices"][0]["message"]["content"]

    # Generate recommendations with rating consideration
    pref_embedding = sbert_model.encode(preference_text, convert_to_tensor=True).to(device)
    similarities = util.cos_sim(pref_embedding, book_embeddings)[0].cpu().numpy()

    # Combine content similarity with rating
    content_similarity = util.cos_sim(pref_embedding, book_embeddings)[0].cpu().numpy()
    normalized_ratings = (df["rating"] / 5).values  # Normalize ratings to 0-1

    # Combined score (70% content similarity, 30% rating)
    combined_score = 0.7 * content_similarity + 0.3 * normalized_ratings
    df["score"] = combined_score

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0)
    df["similarity"] = similarities

    # Get top recommendations
    recommendations = df.sort_values("similarity", ascending=False).head(5)

    # Prepare response
    recs = []
    for _, row in recommendations.iterrows():
        recs.append({
            "title": row["title"],
            "author": row["author"],
            "genres": row["genres"],
            "rating": float(row["rating"]),  # Ensure rating is numeric
            "description": row.get("description", "No description available")
        })

    return jsonify({
        "preferences": preference_text,
        "recommendations": recs,
        "rating_note": "Recommendations consider both your preferences and book ratings"
    })


if __name__ == "__main__":
    app.run(debug=True)