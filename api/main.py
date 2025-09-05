import os
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import json
from datetime import datetime

from src.MultiDocument.multi_doc_handler import DocumentHandler
from src.MultiDocument.multi_doc import DocumentProcessor
from src.evaluation.deep_eval_evaluator import DeepEvalEvaluator, RAGEvaluationMetrics
from exception.custom_exception import DocumentProcessingException
import logging

# Import LangChain caching components
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Placeholder classes for missing modules
class DocumentComparator:
    def __init__(self):
        self.session_id = "default_session"

    def save_uploaded_files(self, file1, file2):
        # Simple placeholder implementation
        return "/tmp/ref.pdf", "/tmp/act.pdf"

    def combine_documents(self):
        # Simple placeholder implementation
        return "Sample combined document text for comparison"

class ChatIngestor:
    def __init__(self, temp_base="data", faiss_base="faiss_index", use_session_dirs=True, session_id=None):
        self.temp_base = temp_base
        self.faiss_base = faiss_base
        self.use_session_dirs = use_session_dirs
        self.session_id = session_id or "default_session"

    def built_retriver(self, files, chunk_size=1000, chunk_overlap=200, k=5, vector_store_choice="existing"):
        try:
            # Create directories if they don't exist
            os.makedirs(self.temp_base, exist_ok=True)
            os.makedirs(self.faiss_base, exist_ok=True)

            # Determine vector store path based on session configuration
            if self.use_session_dirs and self.session_id:
                vector_store_path = os.path.join(self.faiss_base, self.session_id)
            else:
                vector_store_path = self.faiss_base

            # Initialize DocumentProcessor with vector store choice and path
            processor = DocumentProcessor(vector_store_choice=vector_store_choice, vector_store_path=vector_store_path)

            # Process each uploaded file
            for file_adapter in files:
                # Save uploaded file temporarily
                import tempfile
                temp_path = None
                try:
                    # Create temporary file and write content
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_adapter.file.filename}") as temp_file:
                        content = file_adapter.file.file.read()
                        temp_file.write(content)
                        temp_file.flush()
                        temp_path = temp_file.name

                    # Process the document (file handle is now closed)
                    content = processor.process_document(temp_path)
                    if content:
                        processor.add_to_vector_store(file_adapter.file.filename, content)
                        log.info(f"Successfully processed and indexed: {file_adapter.file.filename}")
                    else:
                        log.warning(f"Failed to extract content from: {file_adapter.file.filename}")
                finally:
                    # Clean up temp file with retry logic for Windows
                    if temp_path and os.path.exists(temp_path):
                        for attempt in range(3):  # Retry up to 3 times
                            try:
                                os.remove(temp_path)
                                break  # Success, exit retry loop
                            except PermissionError as e:
                                if attempt < 2:  # Not the last attempt
                                    import time
                                    time.sleep(0.1)  # Wait 100ms before retry
                                    log.warning(f"Retry {attempt + 1} deleting temp file {temp_path}: {e}")
                                else:
                                    log.warning(f"Failed to delete temp file {temp_path} after 3 attempts: {e}")
                                    # On Windows, sometimes files remain locked, let OS clean up later
                            except Exception as e:
                                log.warning(f"Error deleting temp file {temp_path}: {e}")
                                break  # Don't retry for other exceptions

            log.info(f"Index building completed for session: {self.session_id}")
        except Exception as e:
            log.error(f"Error building index: {str(e)}")
            raise

class DocumentAnalyzer:
    def analyze_document(self, text):
        # Enhanced analysis implementation
        try:
            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = len([s for s in text.split('.') if s.strip()])

            # Extract key information
            lines = text.split('\n')
            title = lines[0].strip() if lines else "No title found"

            # Simple keyword extraction (most frequent words)
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1

            top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            keywords = [word for word, freq in top_keywords]

            # Generate a simple summary (first few sentences)
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            summary_sentences = sentences[:3] if len(sentences) >= 3 else sentences
            summary = '. '.join(summary_sentences) + '.' if summary_sentences else "No content to summarize"

            return {
                "title": title,
                "summary": summary,
                "word_count": word_count,
                "character_count": char_count,
                "sentence_count": sentence_count,
                "keywords": keywords,
                "analysis": "Document analysis completed successfully"
            }
        except Exception as e:
            return {
                "summary": f"Document contains {len(text)} characters",
                "word_count": len(text.split()),
                "analysis": f"Basic analysis completed (error in detailed analysis: {str(e)})"
            }

class DocumentComparatorLLM:
    def compare_documents(self, text):
        # Simple placeholder implementation
        import pandas as pd
        return pd.DataFrame({
            "Page": [1, 2],
            "Changes": ["No changes detected", "Minor formatting differences"]
        })

class ConversationalRAG:
    def __init__(self, session_id=None):
        self.session_id = session_id
        self.retriever = None
        self.qa_chain = None

        # Load environment variables for LLM
        import json
        api_keys = json.loads(os.getenv("API_KEYS", "{}"))
        groq_api_key = api_keys.get("GROQ_API_KEY")

        # Initialize embeddings and LLM
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_groq import ChatGroq

        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="compound-beta-mini"
        )

    def load_retriever_from_faiss(self, index_dir, k=5, index_name="index"):
        """Load FAISS retriever from the specified directory."""
        try:
            from langchain_community.vectorstores import FAISS

            # Load the FAISS index
            self.vectorstore = FAISS.load_local(
                index_dir,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

            # Create retriever
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})

            # Create RAG chain
            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate

            rag_prompt = PromptTemplate(
                template="""Answer the question based on the following context:

                Context: {context}

                Question: {question}

                Answer: """,
                input_variables=["context", "question"]
            )

            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.retriever,
                chain_type_kwargs={"prompt": rag_prompt}
            )

            log.info(f"Successfully loaded FAISS retriever from {index_dir}")

        except Exception as e:
            log.error(f"Error loading FAISS retriever from {index_dir}: {str(e)}")
            raise

    def invoke(self, question, chat_history=None):
        """Invoke the RAG chain to answer the question."""
        if self.qa_chain is None:
            raise ValueError("Retriever not loaded. Call load_retriever_from_faiss first.")

        try:
            # Get relevant documents for context
            relevant_docs = self.retriever.get_relevant_documents(question)

            # Use the QA chain to get the answer
            response = self.qa_chain.run(question)

            return {
                "answer": response,
                "sources": [
                    {
                        "source": doc.metadata.get('source', 'Unknown'),
                        "chunk": doc.metadata.get('chunk', 0),
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    }
                    for doc in relevant_docs
                ],
                "session_id": self.session_id
            }

        except Exception as e:
            log.error(f"Error during RAG inference: {str(e)}")
            return {
                "answer": f"I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "session_id": self.session_id,
                "error": str(e)
            }

# Placeholder for missing utilities
class FastAPIFileAdapter:
    def __init__(self, file):
        self.file = file

def read_pdf_via_handler(handler, path):
    # Simple placeholder implementation
    return handler.read_file(path)

FAISS_BASE = os.getenv("FAISS_BASE", "faiss_index")
UPLOAD_BASE = os.getenv("UPLOAD_BASE", "data")
FAISS_INDEX_NAME = os.getenv("FAISS_INDEX_NAME", "index")  # <--- keep consistent with save_local()

# Initialize LangChain in-memory cache for performance optimization
set_llm_cache(InMemoryCache())
log.info("LangChain in-memory cache initialized for performance optimization")

app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp

@app.get("/health")
def health() -> Dict[str, str]:
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}

# ---------- ANALYZE ----------
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Received file for analysis: {file.filename}")
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        try:
            dh = DocumentHandler()
            text = dh.read_file(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)
        log.info("Document analysis complete.")
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Error during document analysis")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

# ---------- COMPARE ----------
@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...), actual: UploadFile = File(...)) -> Any:
    try:
        log.info(f"Comparing files: {reference.filename} vs {actual.filename}")
        dc = DocumentComparator()
        ref_path, act_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, act_path
        combined_text = dc.combine_documents()
        comp = DocumentComparatorLLM()
        df = comp.compare_documents(combined_text)
        log.info("Document comparison completed.")
        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Comparison failed")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

# ---------- CHAT: INDEX ----------
@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    vector_store_choice: str = Form("existing"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        log.info(f"Indexing chat session. Session ID: {session_id}, Files: {[f.filename for f in files]}")
        wrapped = [FastAPIFileAdapter(f) for f in files]
        # this is my main class for storing a data into VDB
        # created a object of ChatIngestor
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,
            faiss_base=FAISS_BASE,
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # NOTE: ensure your ChatIngestor saves with index_name="index" or FAISS_INDEX_NAME
        # e.g., if it calls FAISS.save_local(dir, index_name=FAISS_INDEX_NAME)
        ci.built_retriver(  # if your method name is actually build_retriever, fix it there as well
            wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k, vector_store_choice=vector_store_choice
        )
        log.info(f"Index created successfully for session: {ci.session_id}")
        return {"session_id": ci.session_id, "k": k, "use_session_dirs": use_session_dirs}
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

# ---------- CHAT: QUERY ----------
@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
    enable_evaluation: bool = Form(False),
    ground_truth: Optional[str] = Form(None),
) -> Any:
    try:
        log.info(f"Received chat query: '{question}' | session: {session_id}")
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dirs=True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE  # type: ignore
        if not os.path.isdir(index_dir):
            # Return a placeholder response when no index exists
            log.info(f"No FAISS index found at {index_dir}. Returning placeholder response.")
            return {
                "answer": f"I don't have any documents indexed yet for session '{session_id}'. Please upload and index some documents first using the /chat/index endpoint before querying.",
                "session_id": session_id,
                "k": k,
                "engine": "Placeholder - No Index Available",
                "note": "Upload documents and create an index first"
            }

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir, k=k, index_name=FAISS_INDEX_NAME)  # build retriever + chain
        response = rag.invoke(question, chat_history=[])
        log.info("Chat query handled successfully.")

        # Initialize evaluation results
        evaluation_results = None

        # Perform evaluation if requested
        if enable_evaluation:
            try:
                evaluator = DeepEvalEvaluator()

                # Extract context from sources
                context = [source.get("content", "") for source in response.get("sources", [])]

                # Evaluate the response
                metrics = evaluator.evaluate_rag_response(
                    question=question,
                    answer=response.get("answer", ""),
                    context=context,
                    ground_truth=ground_truth
                )

                # Evaluate against thresholds
                evaluation_results = evaluator.evaluate_with_thresholds(metrics)

                # Save evaluation results
                evaluator.save_evaluation_results(evaluation_results)

                log.info("RAG evaluation completed successfully")

            except Exception as eval_error:
                log.warning(f"Evaluation failed: {str(eval_error)}")
                evaluation_results = {"error": f"Evaluation failed: {str(eval_error)}"}

        # Handle the new response format from ConversationalRAG
        if isinstance(response, dict):
            result = {
                "answer": response.get("answer", "No answer generated"),
                "sources": response.get("sources", []),
                "session_id": response.get("session_id", session_id),
                "k": k,
                "engine": "LCEL-RAG"
            }

            # Add evaluation results if available
            if evaluation_results:
                result["evaluation"] = evaluation_results

            return result
        else:
            # Fallback for string responses
            result = {
                "answer": response,
                "session_id": session_id,
                "k": k,
                "engine": "LCEL-RAG"
            }

            # Add evaluation results if available
            if evaluation_results:
                result["evaluation"] = evaluation_results

            return result
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")

# ---------- EVALUATION ENDPOINTS ----------

@app.post("/evaluation/evaluate")
async def evaluate_rag_response(
    question: str = Form(...),
    answer: str = Form(...),
    context: str = Form(...),  # JSON string of context list
    ground_truth: Optional[str] = Form(None),
    session_id: Optional[str] = Form(None)
) -> Any:
    """Evaluate a RAG response using DeepEval metrics."""
    try:
        log.info(f"Evaluating RAG response for question: '{question[:50]}...'")

        # Parse context from JSON string
        try:
            context_list = eval(context) if context else []
        except:
            context_list = [context] if context else []

        evaluator = DeepEvalEvaluator()

        # Evaluate the response
        metrics = evaluator.evaluate_rag_response(
            question=question,
            answer=answer,
            context=context_list,
            ground_truth=ground_truth
        )

        # Evaluate against thresholds
        evaluation_results = evaluator.evaluate_with_thresholds(metrics)

        # Save evaluation results
        saved_file = evaluator.save_evaluation_results(evaluation_results)

        # Generate evaluation report
        report = evaluator.generate_evaluation_report(evaluation_results)

        log.info("RAG evaluation completed successfully")

        return {
            "metrics": {
                "faithfulness": metrics.faithfulness,
                "answer_relevancy": metrics.answer_relevancy,
                "context_relevancy": metrics.context_relevancy,
                "answer_correctness": metrics.answer_correctness,
                "answer_similarity": metrics.answer_similarity,
                "context_recall": metrics.context_recall,
                "context_precision": metrics.context_precision
            },
            "evaluation_results": evaluation_results,
            "report": report,
            "saved_file": saved_file,
            "session_id": session_id
        }

    except Exception as e:
        log.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")

@app.get("/evaluation/summary")
def get_evaluation_summary() -> Any:
    """Get a summary of all evaluation history."""
    try:
        evaluator = DeepEvalEvaluator()
        summary = evaluator.get_evaluation_summary()

        return {
            "summary": summary,
            "timestamp": str(datetime.now())
        }

    except Exception as e:
        log.exception("Failed to get evaluation summary")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation summary: {e}")

@app.get("/evaluation/results")
def get_evaluation_results() -> Any:
    """Get all saved evaluation result files."""
    try:
        evaluator = DeepEvalEvaluator()
        results_dir = Path(evaluator.results_dir)

        if not results_dir.exists():
            return {"results": [], "message": "No evaluation results found"}

        result_files = []
        for file_path in results_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                result_files.append({
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "timestamp": data.get("timestamp", "Unknown"),
                    "metrics_count": len(data.get("results", {}))
                })
            except Exception as e:
                log.warning(f"Error reading evaluation file {file_path}: {str(e)}")

        return {
            "results": result_files,
            "total_files": len(result_files),
            "results_directory": str(results_dir)
        }

    except Exception as e:
        log.exception("Failed to get evaluation results")
        raise HTTPException(status_code=500, detail=f"Failed to get evaluation results: {e}")

# command for executing the fast api
# uvicorn api.main:app --port 8080 --reload    
#uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload