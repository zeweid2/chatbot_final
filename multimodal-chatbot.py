import streamlit as st
from langchain_openai import ChatOpenAI
import pickle
import os
import numpy as np
from typing import List, Union, Tuple
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

class MultimodalSearchEngine:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.index = None
        self.metadata = None
        
    def process_image(self, image: Image) -> np.ndarray:
        """Process image and get CLIP embedding"""
        inputs = self.clip_processor(images=image, return_tensors="pt")
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach().numpy()
    
    def process_text(self, text: str) -> np.ndarray:
        """Process text and get CLIP embedding"""
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        text_features = self.clip_model.get_text_features(**inputs)
        return text_features.detach().numpy()

    def load_index(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata"""
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[dict]:
        """Search for similar items"""
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results with metadata
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    'index': idx,
                    'distance': float(dist),
                    'metadata': self.metadata[idx]
                })
        return results

# Page config
st.set_page_config(
    page_title="Multimodal Product Search Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session states
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
        search_engine = MultimodalSearchEngine()
        search_engine.load_index(
            'path_to_your_faiss_index.index',
            'path_to_your_metadata.pkl'
        )
        return search_engine
    except Exception as e:
        st.error(f"Error initializing search engine: {str(e)}")
        return None

PROMPT_TEMPLATE = """Based on the retrieved product information below, please provide a detailed response to the user's query:

Retrieved Products:
{context}

User Query: {query}

Guidelines:
1. Provide specific product details from the retrieved information
2. Compare relevant features if multiple products are retrieved
3. Include pricing information if available
4. Suggest similar alternatives if relevant
"""

def process_query(
    query: Union[str, Image.Image],
    search_engine: MultimodalSearchEngine
) -> List[dict]:
    """Process either text or image query and return results"""
    if isinstance(query, str):
        query_embedding = search_engine.process_text(query)
    else:
        query_embedding = search_engine.process_image(query)
    
    return search_engine.search(query_embedding)

def format_results(results: List[dict]) -> str:
    """Format search results for the prompt"""
    formatted = []
    for idx, result in enumerate(results, 1):
        product = result['metadata']
        formatted.append(f"Product {idx}:")
        formatted.append(f"Title: {product.get('title', 'N/A')}")
        formatted.append(f"Price: {product.get('price', 'N/A')}")
        formatted.append(f"Description: {product.get('description', 'N/A')}")
        formatted.append(f"Similarity Score: {1 - result['distance']:.2f}\n")
    return "\n".join(formatted)

# Main app interface
st.title("🔍 Multimodal Product Search Assistant")

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

else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "image" in message:
                st.image(message["image"])

    # Query input
    text_query = st.text_input("Ask about products:")
    uploaded_image = st.file_uploader("Or upload an image:", type=['png', 'jpg', 'jpeg'])
    
    if text_query or uploaded_image:
        query = text_query if text_query else Image.open(uploaded_image)
        
        # Add user message
        user_message = {"role": "user", "content": text_query if text_query else "Image query"}
        if uploaded_image:
            user_message["image"] = uploaded_image
        st.session_state.messages.append(user_message)
        
        with st.chat_message("user"):
            st.write(user_message["content"])
            if "image" in user_message:
                st.image(user_message["image"])

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get relevant products
                results = process_query(query, st.session_state.search_engine)
                context = format_results(results)
                
                # Create message with context and query
                prompt = PROMPT_TEMPLATE.format(
                    context=context,
                    query=text_query if text_query else "Find products similar to the uploaded image"
                )
                
                # Get response from ChatGPT
                chat = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0
                )
                
                response = chat.predict(prompt)
                st.write(response)
                
                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Sidebar controls
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        if st.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.messages = []
            st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by CLIP, FAISS, LangChain, and OpenAI")
