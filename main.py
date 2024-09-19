import os
import json
import numpy as np
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from text_extract import extract_text_from_files
import tiktoken
from utils import count_tokens

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)

import logging
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

def improved_get_relevant_chunks(user_question: str, docsearch: Any, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Retrieves relevant chunks from Pinecone based on user query, combining semantic search with BM25.

    Args:
        user_question (str): The user's query.
        docsearch (Any): The Pinecone vector store.
        top_k (int, optional): The number of top chunks to retrieve. Defaults to 3.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing the 'text' and 'source' of a relevant chunk.
    """
    # 1. Initial Retrieval using Semantic Search (get more results for BM25)
    vector_results = docsearch.similarity_search(user_question, k=top_k * 4)

    # 2. Prepare data for BM25
    all_chunks_text = []
    for result in vector_results:
        chunk_text = result.metadata.get('text') or getattr(result, 'page_content', '')
        if not chunk_text:
            logging.warning(f"Missing text content for result: {result}")
            chunk_text = ""  # Ensure we always have a string, even if empty
        all_chunks_text.append(chunk_text)

    # 3. Apply BM25
    if all_chunks_text:
        bm25 = BM25Okapi([doc.split() for doc in all_chunks_text])
        tokenized_query = user_question.split()
        bm25_scores = bm25.get_scores(tokenized_query)
    else:
        logging.warning("No text content available for BM25 ranking.")
        bm25_scores = [0] * len(vector_results)

    # 4. Combine Semantic Search Scores and BM25 Scores
    combined_scores = []
    for i, result in enumerate(vector_results):
        semantic_score = getattr(result, 'score', 0)
        bm25_score = bm25_scores[i]
        # You can adjust these weights based on performance
        combined_score = (0.7 * semantic_score) + (0.3 * bm25_score)
        combined_scores.append(combined_score)

    # 5. Sort by Combined Scores and Get Top-k
    sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
    top_indices = sorted_indices[:top_k]

    # 6. Extract Text and Source from Top Results
    relevant_chunks = []
    for i in top_indices:
        result = vector_results[i]
        chunk = {
            'text': result.metadata.get('text') or getattr(result, 'page_content', ''),
            'source': result.metadata.get('source', 'Unknown'),
            'score': combined_scores[i]  # Include the combined score for potential debugging
        }
        relevant_chunks.append(chunk)

    return relevant_chunks


def semantic_chunking(text: str) -> List[str]:
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_length = 0
    max_chunk_length = 500

    for sent in sentences:
        if current_length + len(sent) > max_chunk_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sent)
        current_length += len(sent)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def populate_pinecone_index(folder_path, embedding_wrapper, pc, index_name):
    """Processes files, creates chunks, generates embeddings, and upserts to Pinecone.

    Args:
        folder_path (str): The path to the folder containing the files to be processed.
        embedding_wrapper (EmbeddingWrapper): The embedding wrapper object.
        pc (Pinecone): The Pinecone client object.
        index_name (str): The name of the Pinecone index.

    Returns:
        tuple: A tuple containing the Pinecone vector store object and None values for chunks and all_text.

    Raises:
        Exception: If an error occurs during the index population process.
    """
    try:
        all_chunks = []
        all_text = ""

        # Process files and create chunks
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Processing file: {filename}")

            text = extract_text_from_files(file_path)
            if text:
                all_text += text
                chunks = semantic_chunking(text)
                all_chunks.extend([{"text": chunk, "source": filename} for chunk in chunks])
            else:
                logger.warning(f"No text extracted from {filename}")

        # Get the index
        index = pc.Index(index_name)

        # Create embeddings for each chunk
        texts = [chunk["text"] for chunk in all_chunks]
        embeddings = embedding_wrapper.create_embeddings(texts)

        # Prepare data for upsert
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            vector = {
                'id': f'chunk_{i}',
                'values': embedding,
                'metadata': {'text': chunk['text'], 'source': chunk['source']}  # Ensure 'text' key is present
            }
            vectors_to_upsert.append(vector)

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            logger.info(f"Upserted batch of {len(batch)} vectors to Pinecone index")

        logger.info(f"Finished upserting all {len(vectors_to_upsert)} vectors to Pinecone index")

        # Create vector store
        vector_store = PineconeVectorStore(index, embedding_wrapper, "text")
        return vector_store, None, None  # Return None for chunks and all_text

    except Exception as e:
        logger.error(f"Error populating Pinecone index: {str(e)}")
        raise
    
def rewrite_query(client, model, original_query, rewrite_conversation_history):
    system_prompt = '''
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. The RAG has information about Bajaj insurances. 
    Given the original query and the conversation history, rewrite it to be more specific, detailed, and likely to retrieve relevant information. Don't make up information which is not in the question, although using the conversation history you're free to add details if they were mentioned earlier. 
    Consider the context of the conversation when rewriting the query. You're rewriting the queries such that 
    they can be used to do semantic search in a RAG system whose information will be passed on to another LLM 
    for response. Keep this in mind. Not every query needs rewriting; use your judgment. ONLY give the rewritten query as output.
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        *rewrite_conversation_history,
        {"role": "user", "content": f"Original query: {original_query}\n\nRewritten query:"}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=0.5
    )

    rewritten_query = response.choices[0].message.content
    
    rewrite_conversation_history.append({"role": "user", "content": original_query})
    rewrite_conversation_history.append({"role": "assistant", "content": rewritten_query})

    if len(rewrite_conversation_history) > 20:
        rewrite_conversation_history = rewrite_conversation_history[-20:]

    return rewritten_query, rewrite_conversation_history

def chatbot_response(client, model, user_question, relevant_chunks, tools, conversation_history, context=None):
    system_prompt = '''
    You are a helpful and informative assistant specializing in Bajaj Allianz online insurance plans, 
    answering customer questions accurately and comprehensively based solely on extracted text from Bajaj Allianz 
    insurance brochures. You can respond to greetings or answer general questions briefly, 
    but your primary focus is answering insurance-related queries using only the provided text, 
    without making up information or speculating. Your input consists of the customer's 
    question and relevant extracted text. Give direct answers to the questions asked. Your response should not
    sound like you're giving it on the basis of some provided text; it should sound natural.
    
    You also have access to tools that can perform specific actions. Use these tools when appropriate,
    based on the user's query. If a query requires action, use the relevant tool. If it's just an
    informational query, respond based on the relevant chunks provided.

    Remember to consider the context of the previous messages in the conversation when formulating your response.
    '''

    if context:  # If context is provided (bypassed RAG)
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": f"User Question: {user_question}\n\nContext:\n\n{context}"} 
        ]
    else:  # RAG pipeline is active
        messages = [
            {"role": "system", "content": system_prompt},
            *conversation_history,
            {"role": "user", "content": f"User Question: {user_question}\n\nRelevant Excerpts:\n\n{' '.join(relevant_chunks)}"}
        ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        max_tokens=8000,
        temperature=0.4,
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            if function_name == "create_order":
                function_response = create_order(function_args)
            elif function_name == "cancel_order":
                function_response = cancel_order(function_args["order_id"])
            elif function_name == "collect_payment":
                function_response = collect_payment(function_args["amount"], function_args["currency"])
            elif function_name == "view_invoice":
                function_response = view_invoice(function_args["invoice_id"])
            
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })
        
        second_response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return second_response.choices[0].message.content
    else:
        return response_message.content

# Tool functions (API call emulations)
def create_order(order_details: Dict[str, Any]) -> str:
    order_id = "ORD" + ''.join([str(np.random.randint(0, 9)) for _ in range(5)])
    return json.dumps({
        "message": "Order created successfully",
        "order_id": order_id,
        "details": order_details
    })

def cancel_order(order_id: str) -> str:
    return json.dumps({
        "message": f"Order {order_id} cancelled successfully",
        "status": "CANCELLED"
    })

def collect_payment(amount: float, currency: str) -> str:
    fake_link = f"https://bajaj-allianz.com/pay/{amount}-{currency}-" + ''.join([str(np.random.randint(0, 9)) for _ in range(6)])
    return json.dumps({
        "message": "Payment link generated",
        "link": fake_link,
        "amount": amount,
        "currency": currency
    })

def view_invoice(invoice_id: str) -> str:
    fake_link = f"https://bajaj-allianz.com/invoice/{invoice_id}"
    return json.dumps({
        "message": "Invoice retrieved",
        "link": fake_link,
        "invoice_id": invoice_id
    })

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_order",
            "description": "Create a new order for an insurance policy",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_type": {"type": "string", "description": "Type of insurance policy"},
                    "customer_name": {"type": "string", "description": "Name of the customer"},
                    "customer_email": {"type": "string", "description": "Email of the customer"}
                },
                "required": ["policy_type", "customer_name", "customer_email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Cancel an existing order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "ID of the order to be cancelled"}
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "collect_payment",
            "description": "Generate a payment link for policy premium",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to be paid"},"currency": {"type": "string", "description": "Currency of the payment"}
                },
                "required": ["amount", "currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "view_invoice",
            "description": "Retrieve a link to view an invoice",
            "parameters": {
                "type": "object",
                "properties": {
                    "invoice_id": {"type": "string", "description": "ID of the invoice to be viewed"}
                },
                "required": ["invoice_id"]
            }
        }
    }
]