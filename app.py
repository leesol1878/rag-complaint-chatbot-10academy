import gradio as gr
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

print("=" * 50)
print("Loading RAG System...")
print("=" * 50)

# ============================================
# 1. LOAD VECTOR STORE (FULL DATASET)
# ============================================
print("Loading vector store...")
chroma_client = chromadb.PersistentClient(path="vector_store/full_chroma_db")
collection = chroma_client.get_collection(name="complaints_full")
print(f"Vector store loaded: {collection.count():,} chunks")

# ============================================
# 2. LOAD EMBEDDING MODEL
# ============================================
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Embedding model loaded (384-dim)")

# ============================================
# 3. LOAD LLM (Phi-2)
# ============================================
print("Loading LLM (Phi-2)...")
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
print("LLM loaded: Phi-2")
print("=" * 50)

# ============================================
# 4. RETRIEVER FUNCTION
# ============================================
def retrieve_chunks(query, n_results=5):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return {
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0],
        'distances': results['distances'][0]
    }

# ============================================
# 5. CONTEXT FORMATTER
# ============================================
def format_context(retrieved_results, max_chars=2000):
    context_parts = []
    total_chars = 0
    
    for i, (doc, meta) in enumerate(zip(
        retrieved_results['documents'], 
        retrieved_results['metadatas']
    )):
        product = meta.get('product_category', meta.get('product', 'Unknown'))
        issue = meta.get('issue', '')
        
        prefix = f"[Source {i+1}] Product: {product}"
        if issue:
            prefix += f", Issue: {issue}"
        prefix += "\n"
        
        chunk_text = prefix + doc + "\n\n"
        
        if total_chars + len(chunk_text) > max_chars:
            break
        
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)
    
    return "\n".join(context_parts)

# ============================================
# 6. PROMPT CREATOR
# ============================================
def create_prompt(question, context):
    return f"""You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints. Use the following retrieved complaint excerpts to formulate your answer. If the context doesn't contain the answer, state that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

# ============================================
# 7. RAG PIPELINE
# ============================================
def rag_pipeline(question, n_results=5):
    retrieved = retrieve_chunks(question, n_results=n_results)
    context = format_context(retrieved, max_chars=2000)
    prompt = create_prompt(question, context)
    
    response = generator(prompt, max_new_tokens=300)[0]['generated_text']
    answer = response.split("Answer:")[-1].strip()
    
    sources_text = ""
    for i, (doc, meta) in enumerate(zip(retrieved['documents'], retrieved['metadatas'])):
        product = meta.get('product_category', meta.get('product', 'Unknown'))
        issue = meta.get('issue', '')
        company = meta.get('company', '')
        
        sources_text += f"\n--- Source {i+1} ---\n"
        sources_text += f"Product: {product}\n"
        if issue:
            sources_text += f"Issue: {issue}\n"
        if company:
            sources_text += f"Company: {company}\n"
        sources_text += f"Preview: {doc[:200]}...\n"
    
    return {
        'answer': answer,
        'sources': sources_text,
        'num_sources': len(retrieved['documents'])
    }

# ============================================
# 8. SAMPLE QUESTIONS
# ============================================
sample_questions = [
    "What are the most common complaints about credit cards?",
    "Why do customers complain about money transfers?",
    "What issues do customers face with savings accounts?",
    "What problems do people have with personal loans?",
    "Are there any fraud-related complaints?"
]

# ============================================
# 9. GRADIO INTERFACE - SIMPLE VERSION
# ============================================
def ask_question(question):
    """Simple function to process a question and return the answer"""
    if not question or question.strip() == "":
        return "Please enter a question."
    
    try:
        result = rag_pipeline(question, n_results=5)
        
        output = f"**Question:** {question}\n\n"
        output += f"**Answer:**\n{result['answer']}\n\n"
        output += f"**Sources ({result['num_sources']} chunks retrieved):**\n{result['sources']}"
        
        return output
    except Exception as e:
        return f"Error: {str(e)}"

# Create the interface
with gr.Blocks(title="CrediTrust Complaint Analyzer") as demo:
    gr.Markdown("""
    # CrediTrust Complaint Analyzer
    
    Ask questions about customer complaints and get AI-powered answers.
    
    This system analyzes complaints across Credit Cards, Savings Accounts, Money Transfers, and Personal Loans.
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Ask a question",
                placeholder="Type your question here...",
                lines=3
            )
            ask_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear")
        
        with gr.Column(scale=2):
            gr.Markdown("### Sample Questions")
            for q in sample_questions:
                gr.Button(q, size="sm").click(
                    fn=lambda q=q: q,
                    inputs=[],
                    outputs=question_input
                )
    
    output_text = gr.Markdown(label="Response")
    
    # Event handlers
    ask_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=output_text
    )
    
    question_input.submit(
        fn=ask_question,
        inputs=question_input,
        outputs=output_text
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[question_input, output_text]
    )
    
    # Sample question buttons - submit automatically
    for q in sample_questions:
        btn = gr.Button(q, size="sm")
        btn.click(
            fn=lambda q=q: q,
            inputs=[],
            outputs=question_input
        ).then(
            fn=ask_question,
            inputs=question_input,
            outputs=output_text
        )
    
    gr.Markdown("""
    ---
    ### System Status
    - Vector Store: 1,375,327 chunks
    - Model: Phi-2 (2.7B)
    - Embeddings: all-MiniLM-L6-v2
    - Full Dataset: 464K complaints
    
    Built for CrediTrust Financial | 10 Academy AI Mastery | Week 7
    """)

# ============================================
# 10. MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("Starting CrediTrust Complaint Analyzer...")
    print("=" * 50)
    print("Open the link below in your browser:")
    print("   http://127.0.0.1:7860")
    print("=" * 50)
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860
    )