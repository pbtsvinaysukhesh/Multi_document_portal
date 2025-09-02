import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

# Import the FastAPI app
from api.main import app


class TestEvaluationAPI:
    """Test cases for evaluation API endpoints."""

    def setup_method(self):
        """Set up test client before each test method."""
        self.client = TestClient(app)

    def test_evaluate_endpoint_success(self):
        """Test 1: Test successful evaluation endpoint call."""
        test_data = {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of AI that enables computers to learn from data.",
            "context": json.dumps([
                "Machine learning is a method of data analysis.",
                "It is a branch of artificial intelligence."
            ]),
            "ground_truth": "Machine learning is a subset of AI that allows systems to learn from experience.",
            "session_id": "test_session_001"
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "metrics" in data
        assert "evaluation_results" in data
        assert "report" in data
        assert "saved_file" in data
        assert "session_id" in data

        # Check metrics structure
        metrics = data["metrics"]
        required_metrics = [
            "faithfulness", "answer_relevancy", "context_relevancy",
            "answer_correctness", "answer_similarity", "context_recall", "context_precision"
        ]

        for metric in required_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], float)
            assert 0.0 <= metrics[metric] <= 1.0

    def test_evaluate_endpoint_minimal_data(self):
        """Test 2: Test evaluation with minimal required data."""
        test_data = {
            "question": "What is Python?",
            "answer": "Python is a programming language.",
            "context": json.dumps(["Python is a high-level programming language."])
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        assert response.status_code == 200
        data = response.json()

        assert "metrics" in data
        assert "evaluation_results" in data

    def test_evaluate_endpoint_missing_question(self):
        """Test 3: Test evaluation endpoint with missing question."""
        test_data = {
            "answer": "This is an answer.",
            "context": json.dumps(["Some context"])
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        # Should return 422 Unprocessable Entity for missing required field
        assert response.status_code == 422

    def test_evaluate_endpoint_empty_context(self):
        """Test 4: Test evaluation with empty context."""
        test_data = {
            "question": "What is AI?",
            "answer": "AI is artificial intelligence.",
            "context": json.dumps([])
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        assert response.status_code == 200
        data = response.json()

        # Should handle empty context gracefully
        assert "metrics" in data

    def test_evaluation_summary_endpoint(self):
        """Test 5: Test evaluation summary endpoint."""
        response = self.client.get("/evaluation/summary")

        assert response.status_code == 200
        data = response.json()

        assert "summary" in data
        assert "timestamp" in data

        summary = data["summary"]
        assert isinstance(summary, dict)

    def test_evaluation_results_endpoint(self):
        """Test 6: Test evaluation results listing endpoint."""
        response = self.client.get("/evaluation/results")

        assert response.status_code == 200
        data = response.json()

        assert "results" in data
        assert "total_files" in data
        assert "results_directory" in data

        assert isinstance(data["results"], list)
        assert isinstance(data["total_files"], int)

    def test_chat_query_with_evaluation_enabled(self):
        """Test 7: Test chat query endpoint with evaluation enabled."""
        # First, we need to mock the FAISS index check
        with patch('os.path.isdir', return_value=True), \
             patch('api.main.ConversationalRAG') as mock_rag_class:

            # Mock the RAG instance and its methods
            mock_rag_instance = MagicMock()
            mock_rag_class.return_value = mock_rag_instance

            mock_rag_instance.invoke.return_value = {
                "answer": "This is a test answer from the RAG system.",
                "sources": [
                    {
                        "source": "test_doc.pdf",
                        "chunk": 0,
                        "content": "This is test context content for evaluation."
                    }
                ],
                "session_id": "test_session"
            }

            test_data = {
                "question": "What is the capital of France?",
                "session_id": "test_session_001",
                "use_session_dirs": "true",
                "k": "5",
                "enable_evaluation": "true",
                "ground_truth": "The capital of France is Paris."
            }

            response = self.client.post("/chat/query", data=test_data)

            assert response.status_code == 200
            data = response.json()

            # Should include evaluation results when enabled
            assert "evaluation" in data
            assert "answer" in data
            assert "sources" in data

    def test_chat_query_with_evaluation_disabled(self):
        """Test 8: Test chat query endpoint with evaluation disabled."""
        with patch('os.path.isdir', return_value=True), \
             patch('api.main.ConversationalRAG') as mock_rag_class:

            mock_rag_instance = MagicMock()
            mock_rag_class.return_value = mock_rag_instance

            mock_rag_instance.invoke.return_value = {
                "answer": "This is a test answer.",
                "sources": [{"source": "test.pdf", "content": "Test content"}],
                "session_id": "test_session"
            }

            test_data = {
                "question": "What is AI?",
                "session_id": "test_session_001",
                "use_session_dirs": "true",
                "k": "5",
                "enable_evaluation": "false"
            }

            response = self.client.post("/chat/query", data=test_data)

            assert response.status_code == 200
            data = response.json()

            # Should not include evaluation results when disabled
            assert "evaluation" not in data
            assert "answer" in data

    def test_chat_query_no_index(self):
        """Test 9: Test chat query when no FAISS index exists."""
        with patch('os.path.isdir', return_value=False):
            test_data = {
                "question": "What is machine learning?",
                "session_id": "nonexistent_session",
                "use_session_dirs": "true"
            }

            response = self.client.post("/chat/query", data=test_data)

            assert response.status_code == 200
            data = response.json()

            # Should return placeholder response
            assert "note" in data
            assert "Upload documents and create an index first" in data["note"]

    def test_chat_query_missing_session_id(self):
        """Test 10: Test chat query with missing session_id when use_session_dirs=True."""
        test_data = {
            "question": "What is Python?",
            "use_session_dirs": "true",
            "k": "5"
        }

        response = self.client.post("/chat/query", data=test_data)

        assert response.status_code == 400
        data = response.json()

        assert "session_id is required" in data["detail"]

    def test_evaluate_endpoint_invalid_context(self):
        """Test 11: Test evaluation with invalid context format."""
        test_data = {
            "question": "What is testing?",
            "answer": "Testing is checking code quality.",
            "context": "invalid json string",  # Not a valid JSON array
            "ground_truth": "Testing is the process of evaluating software."
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        # Should handle invalid context gracefully or return error
        assert response.status_code in [200, 400, 422]

    def test_evaluate_endpoint_large_context(self):
        """Test 12: Test evaluation with large context."""
        large_context = ["Context chunk " + str(i) * 100 for i in range(20)]

        test_data = {
            "question": "What is the meaning of life?",
            "answer": "The meaning of life is 42.",
            "context": json.dumps(large_context),
            "ground_truth": "The meaning of life is to find happiness."
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        assert response.status_code == 200
        data = response.json()

        assert "metrics" in data

    def test_concurrent_evaluation_requests(self):
        """Test 13: Test multiple concurrent evaluation requests."""
        import threading
        import time

        results = []
        errors = []

        def make_request(request_id):
            try:
                test_data = {
                    "question": f"What is request {request_id}?",
                    "answer": f"Request {request_id} is a test.",
                    "context": json.dumps([f"Context for request {request_id}"]),
                    "session_id": f"session_{request_id}"
                }

                response = self.client.post("/evaluation/evaluate", data=test_data)
                results.append((request_id, response.status_code))
            except Exception as e:
                errors.append((request_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 5
        assert len(errors) == 0

        for request_id, status_code in results:
            assert status_code == 200

    def test_evaluation_with_unicode_content(self):
        """Test 14: Test evaluation with Unicode content."""
        test_data = {
            "question": "What is Schrödinger's cat? 🐱",
            "answer": "Schrödinger's cat is a thought experiment in quantum mechanics. 🧲",
            "context": json.dumps([
                "In quantum mechanics, Schrödinger's cat is a famous thought experiment. 🔬",
                "It illustrates the paradox of quantum superposition. ⚛️"
            ]),
            "ground_truth": "Schrödinger's cat is a quantum physics thought experiment about superposition. 🐈"
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        assert response.status_code == 200
        data = response.json()

        assert "metrics" in data

    def test_evaluation_endpoint_timeout_simulation(self):
        """Test 15: Test evaluation endpoint with simulated timeout (long processing)."""
        # This test would require mocking the evaluation to take longer
        # For now, we'll test with a very long context that might take time to process

        very_long_context = ["Very long context chunk " + "word " * 1000 for _ in range(10)]

        test_data = {
            "question": "What is this long document about?",
            "answer": "This document contains a lot of repeated content for testing purposes.",
            "context": json.dumps(very_long_context),
            "ground_truth": "The document is about testing evaluation with large amounts of text."
        }

        response = self.client.post("/evaluation/evaluate", data=test_data)

        # Should complete successfully despite large input
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
