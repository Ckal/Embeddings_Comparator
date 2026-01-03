import os
import time
import pdfplumber
import docx
import nltk
import gradio as gr
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import (
    OpenAIEmbeddings,
    CohereEmbeddings,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from typing import List, Dict, Any
import pandas as pd

nltk.download('punkt', quiet=True)

FILES_DIR = './files'

MODELS = {
    'HuggingFace': {
        'e5-base-de': "danielheinz/e5-base-sts-en-de",
        'paraphrase-miniLM': "paraphrase-multilingual-MiniLM-L12-v2",
        'paraphrase-mpnet': "paraphrase-multilingual-mpnet-base-v2",
        'gte-large': "gte-large",
        'gbert-base': "gbert-base"
    },
    'OpenAI': {
        'text-embedding-ada-002': "text-embedding-ada-002"
    },
    'Cohere': {
        'embed-multilingual-v2.0': "embed-multilingual-v2.0"
    }
}

class FileHandler:
    @staticmethod
    def extract_text(file_path):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == '.pdf':
            return FileHandler._extract_from_pdf(file_path)
        elif ext == '.docx':
            return FileHandler._extract_from_docx(file_path)
        elif ext == '.txt':
            return FileHandler._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def _extract_from_pdf(file_path):
        with pdfplumber.open(file_path) as pdf:
            return ' '.join([page.extract_text() for page in pdf.pages])

    @staticmethod
    def _extract_from_docx(file_path):
        doc = docx.Document(file_path)
        return ' '.join([para.text for para in doc.paragraphs])

    @staticmethod
    def _extract_from_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

def get_embedding_model(model_type, model_name):
    if model_type == 'HuggingFace':
        return HuggingFaceEmbeddings(model_name=MODELS[model_type][model_name])
    elif model_type == 'OpenAI':
        return OpenAIEmbeddings(model=MODELS[model_type][model_name])
    elif model_type == 'Cohere':
        return CohereEmbeddings(model=MODELS[model_type][model_name])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_text_splitter(split_strategy, chunk_size, overlap_size, custom_separators=None):
    if split_strategy == 'token':
        return TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    elif split_strategy == 'recursive':
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap_size,
            separators=custom_separators or ["\n\n", "\n", " ", ""]
        )
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

def get_vector_store(store_type, texts, embedding_model):
    if store_type == 'FAISS':
        return FAISS.from_texts(texts, embedding_model)
    elif store_type == 'Chroma':
        return Chroma.from_texts(texts, embedding_model)
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

def get_retriever(vector_store, search_type, search_kwargs=None):
    if search_type == 'similarity':
        return vector_store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
    elif search_type == 'mmr':
        return vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    else:
        raise ValueError(f"Unsupported search type: {search_type}")

def process_files(file_path, model_type, model_name, split_strategy, chunk_size, overlap_size, custom_separators):
    if file_path:
        text = FileHandler.extract_text(file_path)
    else:
        text = ""
        for file in os.listdir(FILES_DIR):
            file_path = os.path.join(FILES_DIR, file)
            text += FileHandler.extract_text(file_path)

    text_splitter = get_text_splitter(split_strategy, chunk_size, overlap_size, custom_separators)
    chunks = text_splitter.split_text(text)

    embedding_model = get_embedding_model(model_type, model_name)

    return chunks, embedding_model, len(text.split())

def search_embeddings(chunks, embedding_model, vector_store_type, search_type, query, top_k):
    vector_store = get_vector_store(vector_store_type, chunks, embedding_model)
    retriever = get_retriever(vector_store, search_type, {"k": top_k})

    start_time = time.time()
    results = retriever.get_relevant_documents(query)
    end_time = time.time()

    return results, end_time - start_time, vector_store

def calculate_statistics(results, search_time, vector_store, num_tokens, embedding_model):
    return {
        "num_results": len(results),
        "avg_content_length": sum(len(doc.page_content) for doc in results) / len(results) if results else 0,
        "search_time": search_time,
        "vector_store_size": vector_store._index.ntotal if hasattr(vector_store, '_index') else "N/A",
        "num_documents": len(vector_store.docstore._dict),
        "num_tokens": num_tokens,
        "embedding_vocab_size": embedding_model.client.get_vocab_size() if hasattr(embedding_model, 'client') and hasattr(embedding_model.client, 'get_vocab_size') else "N/A"
    }

def compare_embeddings(file, query, model_types, model_names, split_strategy, chunk_size, overlap_size, custom_separators, vector_store_type, search_type, top_k):
    all_results = []
    all_stats = []
    settings = {
        "split_strategy": split_strategy,
        "chunk_size": chunk_size,
        "overlap_size": overlap_size,
        "custom_separators": custom_separators,
        "vector_store_type": vector_store_type,
        "search_type": search_type,
        "top_k": top_k
    }

    for model_type, model_name in zip(model_types, model_names):
        chunks, embedding_model, num_tokens = process_files(
            file.name if file else None,
            model_type,
            model_name,
            split_strategy,
            chunk_size,
            overlap_size,
            custom_separators.split(',') if custom_separators else None
        )

        results, search_time, vector_store = search_embeddings(
            chunks,
            embedding_model,
            vector_store_type,
            search_type,
            query,
            top_k
        )

        stats = calculate_statistics(results, search_time, vector_store, num_tokens, embedding_model)
        stats["model"] = f"{model_type} - {model_name}"
        stats.update(settings)

        formatted_results = format_results(results, stats)
        all_results.extend(formatted_results)
        all_stats.append(stats)

    results_df = pd.DataFrame(all_results)
    stats_df = pd.DataFrame(all_stats)

    return results_df, stats_df

def format_results(results, stats):
    formatted_results = []
    for doc in results:
        result = {
            "Content": doc.page_content,
            "Model": stats["model"],
            **doc.metadata,
            **{k: v for k, v in stats.items() if k not in ["model"]}
        }
        formatted_results.append(result)
    return formatted_results

# Gradio interface
iface = gr.Interface(
    fn=compare_embeddings,
    inputs=[
        gr.File(label="Upload File (Optional)"),
        gr.Textbox(label="Search Query"),
        gr.CheckboxGroup(choices=list(MODELS.keys()), label="Embedding Model Types", value=["HuggingFace"]),
        gr.CheckboxGroup(choices=[model for models in MODELS.values() for model in models], label="Embedding Models", value=["e5-base-de"]),
        gr.Radio(choices=["token", "recursive"], label="Split Strategy", value="recursive"),
        gr.Slider(100, 1000, step=100, value=500, label="Chunk Size"),
        gr.Slider(0, 100, step=10, value=50, label="Overlap Size"),
        gr.Textbox(label="Custom Split Separators (comma-separated, optional)"),
        gr.Radio(choices=["FAISS", "Chroma"], label="Vector Store Type", value="FAISS"),
        gr.Radio(choices=["similarity", "mmr"], label="Search Type", value="similarity"),
        gr.Slider(1, 10, step=1, value=5, label="Top K")
    ],
    outputs=[
        gr.Dataframe(label="Results"),
        gr.Dataframe(label="Statistics")
    ],
    title="Embedding Comparison Tool",
    description="Compare different embedding models and retrieval strategies",
    examples=[
        [ "files/test.txt", "What is machine learning?", ["HuggingFace"], ["e5-base-de"], "recursive", 500, 50, "", "FAISS", "similarity", 5]
    ],
    flagging_mode="never"
)

# The code remains the same as in the previous artifact, so I'll omit it here for brevity.
# The changes will be in the tutorial_md variable.

tutorial_md = """
# Embedding Comparison Tool Tutorial

This tool allows you to compare different embedding models and retrieval strategies for document search. Before we dive into how to use the tool, let's cover some important concepts.

## What is RAG?

RAG stands for Retrieval-Augmented Generation. It's a technique that combines the strength of large language models with the ability to access and use external knowledge. RAG is particularly useful for:

- Providing up-to-date information
- Answering questions based on specific documents or data sources
- Reducing hallucinations in AI responses
- Customizing AI outputs for specific domains or use cases

RAG is good for applications where you need accurate, context-specific information retrieval combined with natural language generation. This includes chatbots, question-answering systems, and document analysis tools.

## Key Components of RAG

### 1. Document Loading

This is the process of ingesting documents from various sources (PDFs, web pages, databases, etc.) into a format that can be processed by the RAG system. Efficient document loading is crucial for handling large volumes of data.

### 2. Document Splitting

Large documents are often split into smaller chunks for more efficient processing and retrieval. The choice of splitting method can significantly impact the quality of retrieval results.

### 3. Vector Store and Embeddings

Embeddings are dense vector representations of text that capture semantic meaning. A vector store is a database optimized for storing and querying these high-dimensional vectors. Together, they allow for efficient semantic search.

### 4. Retrieval

This is the process of finding the most relevant documents or chunks based on a query. The quality of retrieval directly impacts the final output of the RAG system.

## Why is this important?

Understanding and optimizing each component of the RAG pipeline is crucial because:

1. It affects the accuracy and relevance of the information retrieved.
2. It impacts the speed and efficiency of the system.
3. It determines the scalability of your solution.
4. It influences the overall quality of the generated responses.

## Impact of Parameter Changes

Changes in various parameters can have significant effects:

- **Chunk Size**: Larger chunks provide more context but may reduce precision. Smaller chunks increase precision but may lose context.
- **Overlap**: More overlap can help maintain context between chunks but increases computational load.
- **Embedding Model**: Different models have varying performance across languages and domains.
- **Vector Store**: Affects query speed and the types of searches you can perform.
- **Retrieval Method**: Impacts the diversity and relevance of retrieved documents.

## Detailed Parameter Explanations

### Embedding Model

The embedding model translates text into numerical vectors. The choice of model affects:

- **Language Coverage**: Some models are monolingual, others are multilingual.
- **Domain Specificity**: Models can be general or trained on specific domains (e.g., legal, medical).
- **Vector Dimensions**: Higher dimensions can capture more information but require more storage and computation.

#### Vocabulary Size

The vocab size refers to the number of unique tokens the model recognizes. It's important because:

- It affects the model's ability to handle rare words or specialized terminology.
- Larger vocabs can lead to better performance but require more memory.
- It impacts the model's performance across different languages (larger vocabs are often better for multilingual models).

### Split Strategy

- **Token**: Splits based on a fixed number of tokens. Good for maintaining consistent chunk sizes.
- **Recursive**: Splits based on content, trying to maintain semantic coherence. Better for preserving context.

### Vector Store Type

- **FAISS**: Fast, memory-efficient. Good for large-scale similarity search.
- **Chroma**: Offers additional features like metadata filtering. Good for more complex querying needs.

### Search Type

- **Similarity**: Returns the most similar documents. Fast and straightforward.
- **MMR (Maximum Marginal Relevance)**: Balances relevance with diversity in results. Useful for getting a broader perspective.

## MTEB (Massive Text Embedding Benchmark)

MTEB is a comprehensive benchmark for evaluating text embedding models across a wide range of tasks and languages. It's useful for:

- Comparing the performance of different embedding models.
- Understanding how models perform on specific tasks (e.g., classification, clustering, retrieval).
- Selecting the best model for your specific use case.

### Finding Embeddings on MTEB Leaderboard

To find suitable embeddings using the MTEB leaderboard (https://huggingface.co/spaces/mteb/leaderboard):

1. Look at the "Avg" column for overall performance across all tasks.
2. Check performance on specific task types relevant to your use case (e.g., Retrieval, Classification).
3. Consider the model size and inference speed for your deployment constraints.
4. Look at language-specific scores if you're working with non-English text.
5. Click on model names to get more details and links to the model pages on Hugging Face.

When selecting a model, balance performance with practical considerations like model size, inference speed, and specific task performance relevant to your application.

By understanding these concepts and parameters, you can make informed decisions when using the Embedding Comparison Tool and optimize your RAG system for your specific needs.

## Using the Embedding Comparison Tool

Now that you understand the underlying concepts, here's how to use the tool:

1. **File Upload**: Optionally upload a file (PDF, DOCX, or TXT) or leave it empty to use files in the `./files` directory.

2. **Search Query**: Enter the search query you want to use for retrieving relevant documents.

3. **Embedding Model Types**: Select one or more embedding model types (HuggingFace, OpenAI, Cohere).

4. **Embedding Models**: Choose specific models for each selected model type.

5. **Split Strategy**: Select either 'token' or 'recursive' for text splitting.

6. **Chunk Size**: Set the size of text chunks (100-1000).

7. **Overlap Size**: Set the overlap between chunks (0-100).

8. **Custom Split Separators**: Optionally enter custom separators for text splitting.

9. **Vector Store Type**: Choose between FAISS and Chroma for storing vectors.

10. **Search Type**: Select 'similarity' or 'mmr' (Maximum Marginal Relevance) search.

11. **Top K**: Set the number of top results to retrieve (1-10).

After setting these parameters, click "Submit" to run the comparison. The results will be displayed in two tables:

- **Results**: Shows the retrieved document contents and metadata for each model.
- **Statistics**: Provides performance metrics and settings for each model.

You can download the results as CSV files for further analysis.


## Useful Resources and Links

Here are some valuable resources to help you better understand and work with embeddings, retrieval systems, and natural language processing:

### Embeddings and Vector Databases
- [Understanding Embeddings](https://www.tensorflow.org/text/guide/word_embeddings): A guide by TensorFlow on word embeddings
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss): Facebook AI's vector similarity search library
- [Chroma: The AI-native open-source embedding database](https://www.trychroma.com/): An embedding database designed for AI applications

### Natural Language Processing
- [NLTK (Natural Language Toolkit)](https://www.nltk.org/): A leading platform for building Python programs to work with human language data
- [spaCy](https://spacy.io/): Industrial-strength Natural Language Processing in Python
- [Hugging Face Transformers](https://huggingface.co/transformers/): State-of-the-art Natural Language Processing for PyTorch and TensorFlow 2.0

### Retrieval-Augmented Generation (RAG)
- [LangChain](https://python.langchain.com/docs/get_started/introduction): A framework for developing applications powered by language models
- [OpenAI's RAG Tutorial](https://platform.openai.com/docs/tutorials/web-qa-embeddings): A guide on building a QA system with embeddings

### German Language Processing
- [Kölner Phonetik](https://en.wikipedia.org/wiki/Cologne_phonetics): Information about the Kölner Phonetik algorithm
- [German NLP Resources](https://github.com/adbar/German-NLP): A curated list of open-access resources for German NLP

### Benchmarks and Evaluation
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard): Massive Text Embedding Benchmark leaderboard
- [GLUE Benchmark](https://gluebenchmark.com/): General Language Understanding Evaluation benchmark

### Tools and Libraries
- [Gensim](https://radimrehurek.com/gensim/): Topic modelling for humans
- [Sentence-Transformers](https://www.sbert.net/): A Python framework for state-of-the-art sentence, text and image embeddings


Experiment with different settings to find the best combination for your specific use case!
"""

# The rest of the code remains the same
iface = gr.TabbedInterface(
    [iface, gr.Markdown(tutorial_md)],
    ["Embedding Comparison", "Tutorial"]
)

iface.launch(share=True)