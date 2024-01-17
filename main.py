import ast
import re
import streamlit as st
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

def convert_embedding(x):
    try:
        # Extract numbers using regular expression
        numbers = x.strip('][ \n').split()
        # Convert extracted strings to floats
        return [float(num) for num in numbers]
    except Exception as e:
        print(f"Error converting embedding: {e}")
        return []


st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
    return tokenizer, model

st.cache_data
def load_data():
    # Load your DataFrame here
    df = pd.read_csv('path_to_new_csv_with_embeddings.csv')
    # turn string represtnation of the embeddings column to list of floats
    df['embeddings'] = list(df['embeddings'])
    
    df['embeddings'] = df['embeddings'].apply(convert_embedding)
    # Ensure embeddings in DataFrame are tensors and flattened
    df['embeddings'] = df['embeddings'].apply(lambda x: torch.tensor(x).squeeze().numpy())


    return df

# Function to compute cosine similarity
def cosine_similarity(embedding1, embedding2):
    
    # Convert numpy arrays to PyTorch tensors
    tensor1 = torch.tensor(embedding1)
    tensor2 = torch.tensor(embedding2)
    
    # Apply unsqueeze and compute cosine similarity
    return nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0), dim=1)[0]


# Function to generate an embedding for a given text
def generate_embedding(text, tokenizer, model):
    inputs = tokenizer("query: " + text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze()

# Main app function
def main():
    st.title("Encuentra tu clase Javeriana")

    st.write("Ingresa el tema o nombre de la clase que te interesa y te mostraremos las clases más similares a tu busqueda. ")

    st.caption("*Desarrollado por:* Juan David López")
    st.caption("*Github:* https://github.com/Jl16ExA")
    st.caption("*LinkedIn:* https://www.linkedin.com/in/juan-david-lopez-becerra-5048271bb/")
    st.caption("*Twitter:* https://twitter.com/JLopez_160")

    with st.spinner("Cargando modelo"):
        # Load model and tokenizer
        tokenizer, model = load_model_and_tokenizer()

    with st.spinner("Cargando datos"):
        # Load data
        df = load_data()

    # User inputs
    user_query = st.text_input("Ingresa el tema o nombre de la clase que te interesa", max_chars=512)
   
    if st.button("Search"):
        # Generate embedding for the query string
        query_embedding = generate_embedding(user_query, tokenizer, model)

        
        # Calculate similarity scores
        df['similarity'] = df['embeddings'].apply(lambda x: cosine_similarity(x, query_embedding).item())

       

        # Sort the DataFrame by similarity scores and take top 5
        results = df.sort_values(by='similarity', ascending=False).head(5)

        # Display results
        st.dataframe(results)

# Run the app
if __name__ == "__main__":
    main()
