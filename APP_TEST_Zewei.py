import streamlit as st
import os
import numpy as np
import pandas as pd
import pickle
import torch
from typing import List, Union, Tuple

# Embedding and ML Libraries
from PIL import Image
from transformers import CLIPModel, CLIPImageProcessor
from sentence_transformers import SentenceTransformer
import faiss

# LLM and Chat Libraries
from langchain_openai import ChatOpenAI

class MultimodalSearchEngine:
    def __init__(self, 
                 text_model: str = 'all-mpnet-base-v2', 
                 image_model: str = "openai/clip-vit-base-patch32"):
        """
        Initialize MultimodalSearchEngine with configurable embedding models
        
        Args:
            text_model (str): Sentence Transformer model for text embeddings
            image_model (str): CLIP model for image embeddings
        """
        # Text Embedder
        try:
            self.text_embedder = SentenceTransformer(text_model)
            self.text_embedding_dim = self.text_embedder.get_sentence_embedding_dimension()
        except Exception as e:
            st.error(f"Error loading text embedding model: {e}")
            self.text_embedder = None
            self.text_embedding_dim = 768  # Default fallback
        
        # Image Embedder (CLIP)
        try:
            self.clip_model = CLIPModel.from_pretrained(image_model)
            self.clip_image_processor = CLIPImageProcessor.from_pretrained(image_model)
        except Exception as e:
            st.error(f"Error loading image embedding model: {e}")
            self.clip_model = None
            self.clip_image_processor = None
        
        # Indexes and metadata
        self.text_index = None
        self.image_index = None
        self.metadata_df = None

    def process_text(self, text: str) -> np.ndarray:
        """
        Process text and generate embedding
        
        Args:
            text (str): Input text to embed
        
        Returns:
            np.ndarray: Text embedding
        """
        if self.text_embedder is None:
            raise ValueError("Text embedding model not initialized")
        
        text_features = self.text_embedder.encode(text, convert_to_tensor=False)
        return text_features.reshape(1, -1)

    def process_image(self, image: Image) -> np.ndarray:
        """
        Process image and get CLIP embedding
        
        Args:
            image (Image): Input image to embed
        
        Returns:
            np.ndarray: Image embedding
        """
        if self.clip_model is None or self.clip_image_processor is None:
            raise ValueError("Image embedding model not initialized")
        
        inputs = self.clip_image_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy().reshape(1, 512)

    def load_embeddings(self, 
                        text_path: str, 
                        image_path: str, 
                        metadata_path: str):
        """
        Load text and image embeddings and metadata
        
        Args:
            text_path (str): Path to text embeddings pickle
            image_path (str): Path to image embeddings pickle
            metadata_path (str): Path to metadata CSV
        """
        try:
            # Load text embeddings
            with open(text_path, 'rb') as f:
                text_embeddings = pickle.load(f)
                if isinstance(text_embeddings, list):
                    text_embeddings = np.vstack([emb.reshape(-1, self.text_embedding_dim) for emb in text_embeddings])
                
                # Create FAISS index for text embeddings
                self.text_index = faiss.IndexFlatIP(text_embeddings.shape[1])
                faiss.normalize_L2(text_embeddings.astype(np.float32))
                self.text_index.add(text_embeddings.astype(np.float32))
                
                st.write(f"Loaded text embeddings with shape: {text_embeddings.shape}")
            
            # Load image embeddings
            with open(image_path, 'rb') as f:
                image_embeddings = pickle.load(f)
                if isinstance(image_embeddings, list):
                    image_embeddings = np.vstack([emb.reshape(-1, 512) for emb in image_embeddings])
                
                # Create FAISS index for image embeddings
                self.image_index = faiss.IndexFlatIP(image_embeddings.shape[1])
                faiss.normalize_L2(image_embeddings.astype(np.float32))
                self.image_index.add(image_embeddings.astype(np.float32))
                
                st.write(f"Loaded image embeddings with shape: {image_embeddings.shape}")
            
            # Load metadata CSV
            self.metadata_df = pd.read_csv(metadata_path)
            st.write(f"Loaded metadata with {len(self.metadata_df)} entries")
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            raise

    def search(self, 
               query_embedding: np.ndarray, 
               k: int = 3, 
               mode: str = 'text') -> List[dict]:
        """
        Search for similar items using either text or image index
        
        Args:
            query_embedding (np.ndarray): Embedding to search with
            k (int): Number of results to return
            mode (str): Search mode - 'text' or 'image'
        
        Returns:
            List[dict]: Matching search results with metadata
        """
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Select appropriate index
        index = self.text_index if mode == 'text' else self.image_index
        
        # Perform search
        distances, indices = index.search(query_embedding, k)
        
        # Collect results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata_df):
                metadata_dict = self.metadata_df.iloc[idx].to_dict()
                results.append({
                    'index': idx,
                    'distance': float(dist),
                    'metadata': metadata_dict
                })
        return results

def format_results(results: List[dict]) -> str:
    """
    Format search results for the prompt
    
    Args:
        results (List[dict]): Search results to format
    
    Returns:
        str: Formatted results string
    """
    formatted = []
    for idx, result in enumerate(results, 1):
        product = result['metadata']
        text_desc = product.get('Text_Description', 'N/A')
        parts = text_desc.split('|')
        
        title = parts[0].strip() if parts else 'N/A'
        price = parts[-1].strip() if len(parts) > 1 else 'N/A'
        
        formatted.append(f"Product {idx}:")
        formatted.append(f"Title: {title}")
        formatted.append(f"Price: {price}")
        formatted.append(f"Similarity Score: {1 - result['distance']:.2f}\n")
    
    return "\n".join(formatted)

def display_product_results(results: List[dict]):
    """
    Display product results in a grid with images
    
    Args:
        results (List[dict]): Search results to display
    """
    cols = st.columns(len(results))
    for col, result in zip(cols, results):
        with col:
            product = result['metadata']
            
            # Get the first image URL
            image_urls = product.get('Image_url', '').split('|')
            main_image_url = image_urls[0] if image_urls else None
            
            # Extract title and price
            text_desc = product.get('Text_Description', '')
            parts = text_desc.split('|')
            title = parts[0].strip() if parts else 'N/A'
            price = parts[-1].strip() if len(parts) > 1 else 'N/A'
            
            # Display image
            if main_image_url:
                try:
                    st.image(main_image_url, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            
            # Display product info
            st.markdown(f"**{title}**")
            st.markdown(f"Price: {price}")
            st.markdown(f"Similarity: {1 - result['distance']:.2%}")

# Configure page
st.set_page_config(
    page_title="Advanced Multimodal Product Search",
    page_icon="🔍",
    layout="wide"
)

# Session state management
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None

@st.cache_resource
def initialize_search_engine():
    """Initialize and load the search engine"""
    try:
        # Initialize with optional custom models
        search_engine = MultimodalSearchEngine(
            text_model='all-mpnet-base-v2',  # Customizable text embedding model
            image_model="openai/clip-vit-base-patch32"  # Customizable image embedding model
        )
        
        # Define relative paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        text_path = os.path.join(current_dir, 'text_embeddings.pkl')
        image_path = os.path.join(current_dir, 'image_embeddings.pkl')
        metadata_path = os.path.join(current_dir, 'metadata.csv')
        
        # Load embeddings and metadata
        search_engine.load_embeddings(text_path, image_path, metadata_path)
        return search_engine
    except Exception as e:
        st.error(f"Error initializing search engine: {str(e)}")
        return None

# Prompt template
PROMPT_TEMPLATE = """Based on the retrieved product information, provide a detailed response:

Retrieved Products:
{context}

User Query: {query}

Guidelines:
1. Focus on product features, price, and relevance
2. Compare prices and features
3. Make specific recommendations
4. Explain why products are similar
5. Include price comparisons"""

def main():
    st.title("🔍 Advanced Multimodal Product Search")

    # Initialize search engine
    if st.session_state.search_engine is None:
        with st.spinner("Initializing search engine..."):
            st.session_state.search_engine = initialize_search_engine()
            if st.session_state.search_engine is None:
                st.error("Failed to initialize search engine")
                st.stop()

    # API Key input
    if not st.session_state.api_key:
        with st.form("api_key_form"):
            api_key = st.text_input("OpenAI API Key:", type="password")
            submitted = st.form_submit_button("Submit")
            
            if submitted and api_key.startswith('sk-'):
                st.session_state.api_key = api_key
                os.environ['OPENAI_API_KEY'] = api_key
                st.rerun()
            elif submitted:
                st.error("Please enter a valid OpenAI API key")

    # Rest of the Streamlit app logic remains the same as in the previous implementation
    # (Keep the chat interface, query processing, and result display code from the previous script)

if __name__ == "__main__":
    main()
