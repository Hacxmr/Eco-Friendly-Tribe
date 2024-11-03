import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
url = "https://huggingface.co/datasets/infinite-dataset-hub/FabricFrontiers/resolve/main/data.csv"
df = pd.read_csv(url)

# Preprocess the data
encoder = OneHotEncoder()
encoded_labels = encoder.fit_transform(df[['label']]).toarray()

# Vectorize 'title' and 'description'
text_data = df['title'] + " " + df['description']
vectorizer = TfidfVectorizer(max_features=100)
text_vectors = vectorizer.fit_transform(text_data).toarray()

# Combine into a feature matrix
df_prepared = np.hstack([encoded_labels, text_vectors])

# Set up FAISS
vector_dim = df_prepared.shape[1]
index = faiss.IndexFlatL2(vector_dim)
index.add(df_prepared.astype('float32'))

# Define function to query similar items
def query_similar_items(input_vector, top_k=5):
    input_vector = np.array(input_vector).astype('float32').reshape(1, -1)
    distances, indices = index.search(input_vector, top_k + 1)
    recommendations = indices[0][1:]  # Skip the first match (itself)
    return recommendations

# Streamlit frontend

# Title and Instructions
st.title("Fabric Recommendations")
st.write("Select a fabric to get similar recommendations.")

# Dropdown for fabric selection
fabric_names = df['title'].unique()  # Get unique fabric titles for dropdown
selected_fabric = st.selectbox("Select Fabric:", fabric_names)

# Check if the fabric is selected
if selected_fabric:
    # Find the index of the selected fabric
    selected_item_index = df[df['title'] == selected_fabric].index[0]
    selected_item = df.iloc[selected_item_index]

    # Display selected item details
    st.subheader("Selected Item:")
    st.write("**Title**:", selected_item['title'])
    st.write("**Description**:", selected_item['description'])
    st.write("**Source**:", selected_item['source'])
    st.write("**Label**:", selected_item['label'])

    # Get the feature vector of the selected item
    example_vector = df_prepared[selected_item_index]

    # Fetch and display similar items
    st.subheader("Recommended Items:")
    similar_items = query_similar_items(example_vector)
    for idx in similar_items:
        recommended_item = df.iloc[idx]
        st.write("**Title**:", recommended_item['title'])
        st.write("**Description**:", recommended_item['description'])
        st.write("**Source**:", recommended_item['source'])
        st.write("**Label**:", recommended_item['label'])
        st.write("---")
