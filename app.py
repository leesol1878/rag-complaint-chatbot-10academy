"""
RAG-Powered Complaint Analysis Chatbot
CrediTrust Financial - 10 Academy AI Mastery
"""

import gradio as gr
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

print("=" * 50)
print("🚀 Loading RAG System...")
print("=" * 50)

# ============================================
# 1. LOAD VECTOR STORE
# ============================================
print("📚 Loading vector store...")
chroma_client = chromadb.PersistentClient(path="vector_store/chroma_db")
collection = chroma_client.get_collection(name="complaints")
print(f"✅ Vector store loaded: {collection.count()} chunks")

# ============================================
# 2. LOAD EMBEDDING MODEL
# ============================================
print("🧠 Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Embedding model loaded (384-dim)")

# ============================================
# 3. LOAD LLM (Phi-2)
# ============================================
print("🤖 Loading LLM (Phi-2)...")
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
print("✅ LLM loaded: Phi-2")
print("=" * 50)

# ============================================
# 4. RETRIEVER FUNCTION
# ============================================
def retrieve_chunks(query, n_results=5):
    """Retrieve relevant chunks from vector store"""
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
    """Format retrieved chunks into context"""
    context_parts = []
    total_chars = 0
    
    for i, (doc, meta) in enumerate(zip(
        retrieved_results['documents'], 
        retrieved_results['metadatas']
    )):
        product = meta.get('product', 'Unknown')
        issue = meta.get('issue', '')
        prefix = f"[Source {i+1}] Product: {product}"
        if issue:
            prefix += f", Issue: {issue}"
        prefix += "\n"
        
        chunk_text = prefix + doc + "\n"
        
        if total_chars + len(chunk_text) > max_chars:
            break
        
        context_parts.append(chunk_text)
        total_chars += len(chunk_text)
    
    return "\n".join(context_parts)

# ============================================
# 6. PROMPT CREATOR
# ============================================
def create_prompt(question, context):
    """Create prompt for LLM"""
    return f"""You are a financial analyst assistant for CrediTrust Financial. Your task is to answer questions about customer complaints.

Instructions:
1. Use ONLY the provided context to answer the question
2. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
3. Be concise and specific
4. Cite the source product when relevant
5. Do not make up information

Context:
{context}

Question: {question}

Answer:"""

# ============================================
# 7. RAG PIPELINE
# ============================================
def rag_pipeline(question, n_results=5):
    """Complete RAG pipeline"""
    # Retrieve
    retrieved = retrieve_chunks(question, n_results=n_results)
    
    # Format context
    context = format_context(retrieved, max_chars=2000)
    
    # Create prompt
    prompt = create_prompt(question, context)
    
    # Generate answer
    response = generator(prompt, max_new_tokens=300)[0]['generated_text']
    answer = response.split("Answer:")[-1].strip()
    
    # Format sources for display
    sources_text = ""
    for i, (doc, meta) in enumerate(zip(retrieved['documents'], retrieved['metadatas'])):
        product = meta.get('product', 'Unknown')
        issue = meta.get('issue', '')
        sources_text += f"\n📄 **Source {i+1}**\n"
        sources_text += f"   Product: {product}\n"
        if issue:
            sources_text += f"   Issue: {issue}\n"
        sources_text += f"   Preview: {doc[:200]}...\n"
    
    return {
        'answer': answer,
        'sources': sources_text,
        'context': context,
        'num_sources': len(retrieved['documents'])
    }

# ============================================
# 8. GRADIO INTERFACE FUNCTIONS
# ============================================
def chat_function(message, history):
    """Process user message and return response"""
    if not message or message.strip() == "":
        return "Please enter a question."
    
    try:
        # Run RAG pipeline
        result = rag_pipeline(message, n_results=5)
        
        # Format response
        response = f"### 💡 Answer\n\n{result['answer']}\n\n"
        response += f"### 📚 Sources ({result['num_sources']} chunks retrieved)\n\n"
        response += result['sources']
        
        return response
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ============================================
# 9. SAMPLE QUESTIONS
# ============================================
sample_questions = [
    "Why are customers unhappy with credit cards?",
    "What are the main issues with savings accounts?",
    "Why do customers complain about money transfers?",
    "What problems do people have with personal loans?",
    "Are there any fraud-related complaints?"
]

# ============================================
# 10. GRADIO INTERFACE
# ============================================
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="CrediTrust Complaint Analyzer",
        theme=gr.themes.Soft(),
        css=""" 
        .gradio-container { max-width: 900px; margin: auto; }
        h1 { text-align: center; color: #1a365d; }
        .footer { text-align: center; margin-top: 20px; color: #666; }
        """
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🏦 CrediTrust Complaint Analyzer
        
        **Ask questions about customer complaints and get AI-powered answers.**
        
        *This system analyzes complaints across Credit Cards, Savings Accounts, Money Transfers, and Personal Loans.*
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="💬 Conversation",
                    height=400,
                    show_copy_button=True
                )
                
                msg = gr.Textbox(
                    label="Ask a question",
                    placeholder="Type your question here...",
                    lines=2,
                    show_label=False
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Ask", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
            
            with gr.Column(scale=1):
                # Sidebar
                gr.Markdown("### 📋 Sample Questions")
                
                for q in sample_questions:
                    gr.Button(q, size="sm").click(
                        fn=lambda q=q: q,
                        outputs=msg
                    ).then(
                        fn=chat_function,
                        inputs=[msg, None],
                        outputs=chatbot
                    )
                
                gr.Markdown("""
                ### 📊 System Status
                - ✅ Vector Store: 28,866 chunks
                - ✅ Model: Phi-2 (2.7B)
                - ✅ Embeddings: all-MiniLM-L6-v2
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <div class="footer">
        Built for CrediTrust Financial • 10 Academy AI Mastery • Week 7
        </div>
        """)
        
        # ============================================
        # EVENT HANDLERS
        # ============================================
        def respond(message, chat_history):
            """Handle user message and update chat"""
            if not message or message.strip() == "":
                return "", chat_history
            
            # Add user message
            chat_history.append([message, None])
            
            # Get response
            response = chat_function(message, chat_history)
            
            # Update chat
            chat_history[-1][1] = response
            
            return "", chat_history
        
        # Submit handlers
        submit_btn.click(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot, msg]
        )
    
    return demo

# ============================================
# 11. MAIN
# ============================================
if __name__ == "__main__":
    print("=" * 50)
    print("🚀 Starting Gradio Interface...")
    print("=" * 50)
    print("🌐 Open the link below in your browser:")
    print("   http://127.0.0.1:7860")
    print("=" * 50)
    
    demo = create_interface()
    demo.launch(
        share=False,  # Set to True for public link
        server_name="127.0.0.1",
        server_port=7860
    )