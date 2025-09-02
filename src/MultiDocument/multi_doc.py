from .multi_doc_handler import DocumentHandler
from pathlib import Path
from typing import List, Optional, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import json
import time


class DocumentProcessor:
    def __init__(self, vector_store_choice="existing", vector_store_path="vector_store"):
        # Load environment variables
        load_dotenv()
        api_keys = json.loads(os.getenv("API_KEYS"))
        groq_api_key = api_keys.get("GROQ_API_KEY")

        self.handler = DocumentHandler()

        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        # Initialize Groq LLM for RAG with updated model
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="compound-beta-mini"  # Updated model name
        )

        # Handle vector store initialization
        self.vector_store_path = vector_store_path
        self.vector_store = None
        self.vector_store_choice = vector_store_choice

        # Check if vector store exists and load it if available
        if os.path.exists(self.vector_store_path):
            if vector_store_choice == "existing":
                try:
                    self.vector_store = FAISS.load_local(
                        self.vector_store_path,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    print("Loaded existing vector store...")
                except Exception as e:
                    print(f"Error loading existing vector store: {e}")
                    print("Creating new vector store...")
                    self.vector_store = None
            else:  # vector_store_choice == "new"
                print("Creating new vector store as requested...")
                # Backup existing vector store
                backup_path = f"{self.vector_store_path}_backup_{int(time.time())}"
                os.rename(self.vector_store_path, backup_path)
                print(f"Existing vector store backed up to: {backup_path}")
                self.vector_store = None
        else:
            print("No existing vector store found. Will create new one when documents are processed.")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        
        # Initialize RAG prompt template
        self.rag_prompt = PromptTemplate(
            template="""Answer the question based on the following context:
            
            Context: {context}
            
            Question: {question}
            
            Answer: """,
            input_variables=["context", "question"]
        )


    def get_supported_files(self, folder_path: str) -> List[Path]:
        """Get all supported document files from the specified folder."""
        supported_extensions = {'.docx', '.pdf', '.txt', '.xlsx', 
                              '.csv', '.pptx', '.ppt', '.md', '.db'}
        
        folder = Path(folder_path)
        if not folder.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
            
        return [f for f in folder.glob('*') 
                if f.is_file() and f.suffix.lower() in supported_extensions]

    def add_to_vector_store(self, filename: str, content: str):
        """Split content and add to vector store."""
        # Split text into chunks
        chunks = self.text_splitter.split_text(content)
        
        # Add metadata to chunks
        metadatas = [{"source": filename, "chunk": i} for i in range(len(chunks))]
        
        # Initialize vector store if needed
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings,
                metadatas=metadatas
            )
        else:
            self.vector_store.add_texts(
                texts=chunks,
                metadatas=metadatas
            )
        
        # Save vector store locally
        self.vector_store.save_local(self.vector_store_path)
        print(f"Added {len(chunks)} chunks from {filename} to vector store")

    def process_document(self, file_path: str) -> Optional[str]:
        """Process a single document and extract its content."""
        try:
            content = self.handler.read_file(file_path)
            print(f"Successfully processed {Path(file_path).name}")
            return content
        except Exception as e:
            print(f"Error processing {Path(file_path).name}: {str(e)}")
            return None

    def process_folder(self, folder_path: str):
        """Process all supported documents and add to vector store."""
        try:
            files = self.get_supported_files(folder_path)
            if not files:
                print(f"No supported documents found in {folder_path}")
                return
                
            print(f"Found {len(files)} supported documents")
            for file_path in files:
                content = self.process_document(str(file_path))
                if content:
                    self.add_to_vector_store(file_path.name, content)
                    
        except Exception as e:
            print(f"Error processing folder {folder_path}: {str(e)}")

    def query_vector_store(self, query: str, k: int = 5):
        """Query using RAG with Groq."""
        if self.vector_store is None:
            print("No documents have been processed yet.")
            return None
        
        try:
            rag_chain = self.setup_rag_chain()
            # Replace deprecated run() with invoke()
            response = rag_chain.invoke({"query": query})
            
            # Get relevant chunks for context
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            return {
                "answer": response["result"],  # Updated to get result from invoke response
                "sources": [
                    {
                        "source": doc.metadata['source'],
                        "chunk": doc.metadata['chunk'],
                        "content": doc.page_content[:200] + "..."
                    }
                    for doc in relevant_docs
                ]
            }
        except Exception as e:
            print(f"Error during RAG inference: {str(e)}")
            return None
        
    def setup_rag_chain(self):
        """Setup RAG chain with Groq LLM and vector store."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Process documents first.")
            
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": self.rag_prompt
            }
        )

    def query_vector_store(self, query: str, k: int = 5):
        """Query using RAG with Groq."""
        if self.vector_store is None:
            print("No documents have been processed yet.")
            return []
        
        try:
            rag_chain = self.setup_rag_chain()
            response = rag_chain.run(query)
            
            # Get relevant chunks for context
            relevant_docs = self.vector_store.similarity_search(query, k=k)
            
            return {
                "answer": response,
                "sources": [
                    {
                        "source": doc.metadata['source'],
                        "chunk": doc.metadata['chunk'],
                        "content": doc.page_content[:200] + "..."
                    }
                    for doc in relevant_docs
                ]
            }
        except Exception as e:
            print(f"Error during RAG inference: {str(e)}")
            return None


