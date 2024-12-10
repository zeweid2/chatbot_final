import streamlit as st
from langchain_openai import ChatOpenAI
import pickle
import os
import numpy as np
from typing import List

# Page config
st.set_page_config(
    page_title="MADS Program Chat Assistant",
    page_icon="🎓",
    layout="wide"
)

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None

@st.cache_resource
def load_preprocessed_data():
    """Load the preprocessed data from pickle file"""
    try:
        with open('processed_data.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading preprocessed data: {str(e)}")
        return None

def find_relevant_context(query: str, preprocessed_data: dict, k: int = 3) -> str:
    """Find most relevant texts using cosine similarity"""
    from langchain_openai import OpenAIEmbeddings
    
    # Get query embedding
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    
    # Calculate similarities
    similarities = [
        np.dot(query_embedding, doc_embedding) / 
        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        for doc_embedding in preprocessed_data['embeddings']
    ]
    
    # Get top k most similar texts
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    relevant_texts = [preprocessed_data['texts'][i] for i in top_k_indices]
    
    return "\n\n".join(relevant_texts)

# Your custom prompt template
PROMPT_TEMPLATE = """Given the context below, please provide an accurate and detailed response to the user's inquiry about the MS in Applied Data Science program at the University of Chicago:
{context}

Question: {question}
Guidelines:
1. Use only information from the context provided.
2. Craft a detailed response that addresses the inquiry directly.
3. Use bullet points points if applicable.
4. Include a relevant URL if it directly supports the answer, please only select from the following urls: https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/faqs/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/tuition-fees-aid/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/in-person-program/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/course-progressions/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/capstone-projects/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/instructors-staff/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/online-program/%20
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/our-students/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/how-to-apply/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/online-program/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/career-outcomes/
https://datascience.uchicago.edu/education/masters-programs/ms-in-applied-data-science/events-deadlines/
"""

# Main app interface
st.title("🎓 MADS Program Chat Assistant")

# Check for preprocessed data
if not os.path.exists('processed_data.pkl'):
    st.error("""
    Preprocessed data not found! Please run preprocess.py first.
    Check the README.md for instructions.
    """)
    st.stop()

# Load preprocessed data
if st.session_state.preprocessed_data is None:
    with st.spinner("Loading knowledge base..."):
        st.session_state.preprocessed_data = load_preprocessed_data()
        if st.session_state.preprocessed_data is None:
            st.error("Failed to load knowledge base")
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

    # Chat input
    if question := st.chat_input("Ask about the MADS program"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get relevant context
                relevant_context = find_relevant_context(
                    question, 
                    st.session_state.preprocessed_data
                )
                
                # Create message with context and question
                prompt = PROMPT_TEMPLATE.format(
                    context=relevant_context,
                    question=question
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
st.caption("Powered by LangChain and OpenAI")
