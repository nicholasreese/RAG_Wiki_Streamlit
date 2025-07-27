# RAG Wiki Streamlit

A Retrieval-Augmented Generation (RAG) application that uses Wikipedia articles to answer questions. Built with Streamlit and LlamaIndex.

## Features

- Search and retrieve information from Wikipedia articles
- Use OpenAI's GPT models for question answering
- Interactive Streamlit web interface
- Persistent vector index for faster queries

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd RAG_Wiki_Streamlit
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage

1. Open the application in your browser
2. Enter a question in the text input
3. Click "Submit Query" to get an answer
4. View the retrieved context from Wikipedia articles

## Configuration

The application comes pre-configured with Wikipedia articles about:
- Python Programming Language
- Artificial Intelligence
- Machine Learning
- Data Science
- Deep Learning
- Neural Networks
- Computer Vision
- Natural Language Processing
- Robotics
- Blockchain

You can modify the `PAGES` list in `main.py` to include different Wikipedia articles.

## Notes

- The first run will take longer as it downloads and indexes the Wikipedia articles
- Subsequent runs will be faster as the index is persisted locally
- Make sure you have sufficient OpenAI API credits for embeddings and completions 