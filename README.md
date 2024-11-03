# Fabric Recommendation System

This project is a fabric recommendation system that suggests similar fabrics based on an initial selection. Using FAISS (Facebook AI Similarity Search), it indexes the fabric data and finds items with similar attributes. The application is built using Streamlit for the frontend, allowing users to interactively choose a fabric and view similar recommendations.

## Features

- **Fabric Selection**: Choose a fabric from a dropdown menu to find similar fabrics.
- **Text Vectorization**: Uses TF-IDF to transform fabric titles and descriptions into numerical vectors.
- **Label Encoding**: One-hot encoding of categorical labels.
- **Similarity Search**: Utilizes FAISS to find and recommend fabrics similar to the selected one.

## Demo

![Screenshot](screenshot.png) <!-- Optional: Add a screenshot of the app here -->

## Dataset

The dataset used is [FabricFrontiers](https://huggingface.co/datasets/infinite-dataset-hub/FabricFrontiers), a CSV file containing columns like `idx`, `title`, `description`, `source`, and `label`.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/fabric-recommendation-system.git
    cd fabric-recommendation-system
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:

```bash
streamlit run streamlit_app.py

