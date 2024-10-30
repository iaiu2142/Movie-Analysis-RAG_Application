# Movie-Analysis-RAG_Application
## Technologies: Python | RAG | GROQ API | Llama Model | Kaggle Dataset
![image](https://github.com/user-attachments/assets/e52c9c8a-9ebd-40d7-9d78-ea33d276ca40)

# Overview
This project uses Retrieval-Augmented Generation (RAG) with the Llama model to analyze and summarize movie scenes. Developed in collaboration with PakAngels, USA, the application processes a pre-defined Kaggle dataset and employs the GROQ API for efficient real-time scene understanding.
# Features:
- AI-Powered Summarization: Automatically generates concise summaries of movie scenes.
- Pre-defined Dataset: Utilizes movie data scraped from Kaggle for accurate analysis.
- High-Efficiency Processing: Powered by GROQ API and Llama model for real-time results.
# Requirements:
To run this project locally, ensure you have the following dependencies installed:
```code
pip install transformers
pip install datasets
pip install groqflow
pip install flask
```
# 1. Setup Instructions
## Clone the Repository:
Clone the repository to your local machine:
```code
git clone <your-github-repo-url>
cd <your-repo-directory>
```
## 2. Install Dependencies:
Install all necessary dependencies using pip:
```code
pip install -r requirements.txt
```
## 3. Accessing the Pre-defined Dataset:
Download the movie dataset from Kaggle:

```code
export GROQ_API_KEY='your_groq_api_key'
```
## 4. Run the Application:
Once the setup is complete, you can run the app as follows:
```code
python app.py
```
This will launch the application, allowing you to input movie scenes and receive AI-powered scene summaries.
# Code Overview
## Using  **LLAMA Model**  and  **API Key**  for Chat with Application
The application's core is based on the Llama model integrated with RAG. Below is a snippet that loads the model and sets up the pipeline:
```code
import os
from groq import Groq
client = Groq(
    api_key= API_KEY,)
chat_completion = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "Explain the importance of fast language models",
        }
    ],
    model="llama3-8b-8192",
)
print(chat_completion.choices[0].message.content)
```
# Reading the Dataset with Pandas:
```code
import pandas as pd
# Load your dataset into a DataFrame
pd = pd.read_csv("/content/Hydra-Movie-Scrape - Hydra-Movie-Scrape (1).csv")
# Use the head() method on the DataFrame to display the first 6 rows
df = pd.head(6)
print(df)
```
# Application Generation Process:
```code
import pandas as pd
import numpy as np
from groq import Groq
import faiss
from sentence_transformers import SentenceTransformer

# Step 1: Initialize the Groq API client with your API key
client = Groq(api_key=API_KEY)

# Step 2: Load your dataset
data = pd.read_csv("/content/Hydra-Movie-Scrape - Hydra-Movie-Scrape (1).csv")

# Step 3: Handle missing values in the "Summary" column by replacing NaN with empty strings
data['Summary'].fillna('', inplace=True)

# Step 4: Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Step 5: Embed the "Summary" column
embeddings = embedder.encode(data['Summary'].tolist(), show_progress_bar=True)

# Step 6: Store the embeddings as a list of arrays in the DataFrame
data['summary_embeddings'] = list(embeddings)

# Step 7: Initialize FAISS index for vector search
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))  # Add embeddings to FAISS

# Step 8: Define a function to search for similar summaries using FAISS
def search_similar_summaries(query, top_n=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_n)
    return data.iloc[indices[0]]

# Step 9: Define a function to generate the augmented response using the Groq API
def generate_response(user_query):
    # Retrieve similar movie summaries from the dataset
    retrieved_data = search_similar_summaries(user_query, top_n=5)

    # Create context from the retrieved data to pass as input to the Groq API
    context = ""
    for idx, row in retrieved_data.iterrows():
        context += f"Title: {row['Title']}\nSummary: {row['Summary']}\nGenres: {row['Genres']}\nDirector: {row['Director']}\n\n"

    # Construct the final prompt for Groq's language model
    prompt = f"The user asked: {user_query}\nHere are some relevant movies:\n{context}"

    # Use the Groq API to get a completion from the language model
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama3-8b-8192",
    )
    # Access the generated response using dot notation
    generated_text = chat_completion.choices[0].message.content
    return generated_text

# Step 10: Example usage of the RAG application
user_query = "Man of steel"
response = generate_response(user_query)

# Output the generated response
print(response)
```
## YouTube: https://www.youtube.com/live/8aQSUfHCg0A?si=NdfjB-PaH-qpGNG_&t=40
