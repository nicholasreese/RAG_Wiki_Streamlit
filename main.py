# A RAG application that looks up Wikipedia  

import os
import nltk
nltk.download('punkt')

import streamlit as st
from dotenv import load_dotenv

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage

load_dotenv()

# Check for required environment variables
if not os.getenv("OPENAI_API_KEY"):
    st.error("Please set OPENAI_API_KEY in your .env file")
    st.stop()

INDEX_DIR = "index"
PAGES = [
    "Python Programming Language",
    "Artificial Intelligence", 
    "Machine Learning", 
    "Data Science",
    "Deep Learning",
    "Neural Networks",
    "Computer Vision",
    "Natural Language Processing",
    "Robotics",
    "Blockchain",
]



@st.cache_resource
def get_index():
    try:
        if os.path.isdir(INDEX_DIR):
            storage = StorageContext.from_defaults(persist_dir=INDEX_DIR)
            return load_index_from_storage(storage)
        
        docs = WikipediaReader().load_data(pages=PAGES, auto_suggest=False)
        embedding_model = OpenAIEmbedding(model='text-embedding-3-small')
        index = VectorStoreIndex.from_documents(docs, embed_model=embedding_model)
        index.storage_context.persist(persist_dir=INDEX_DIR)
        
        return index
    except Exception as e:
        st.error(f"Error creating index: {e}")
        return None

@st.cache_resource
def get_query_engine():
    index = get_index()
    
    if index is None:
        return None
    
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    
    return index.as_query_engine(llm=llm, similarity_top_k=3)

def main():
    st.title("Wikipedia RAG")
    
    # Add a button to clear the index if there are issues
    if st.button("Clear Index (if corrupted)"):
        import shutil
        if os.path.exists(INDEX_DIR):
            shutil.rmtree(INDEX_DIR)
            st.success("Index cleared! Please refresh the page.")
            st.stop()
    
    question = st.text_input("Enter a question")
    
    if st.button("Submit Query") and question:
        try:
            with st.spinner("Searching..."):
                qa = get_query_engine()
                if qa is None:
                    st.error("Failed to initialize query engine")
                    return
                response = qa.query(question)
                
            st.subheader("Answer")
            st.write(response.response)
            
            st.subheader("Retrieved Context")
            
            for src in response.source_nodes:
                st.markdown(src.node.get_content())
        except Exception as e:
            st.error(f"Error processing query: {e}")       
            
if __name__ == "__main__":
    main()

