import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

# Import the evaluation components
from src.evaluation.deep_eval_evaluator import (
    DeepEvalEvaluator,
    RAGEvaluationMetrics,
    EvaluationResult
)


class TestDeepEvalEvaluator:
    """Test cases for DeepEvalEvaluator class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.evaluator = DeepEvalEvaluator(results_dir="test_evaluation_results")
        self.sample_question = "What is machine learning?"
        self.sample_answer = "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed."
        self.sample_context = [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "It is a branch of artificial intelligence based on the idea that systems can learn from data.",
            "Machine learning algorithms build a model based on training data."
        ]
        self.sample_ground_truth = "Machine learning is a subset of AI that allows systems to automatically learn and improve from experience without being explicitly programmed."

    def teardown_method(self):
        """Clean up after each test method."""
        # Remove test results directory if it exists
        test_dir = Path("test_evaluation_results")
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

    def test_evaluator_initialization(self):
        """Test 1: Verify evaluator initializes with correct default values."""
        assert self.evaluator.results_dir == Path("test_evaluation_results")
        assert self.evaluator.thresholds["faithfulness"] == 0.7
        assert self.evaluator.thresholds["answer_relevancy"] == 0.7
        assert len(self.evaluator.evaluation_history) == 0

    def test_calculate_faithfulness(self):
        """Test 2: Test faithfulness calculation."""
        faithfulness = self.evaluator._calculate_faithfulness(
            self.sample_answer, self.sample_context
        )

        assert isinstance(faithfulness, float)
        assert 0.0 <= faithfulness <= 1.0

        # Test with empty context
        empty_faithfulness = self.evaluator._calculate_faithfulness(self.sample_answer, [])
        assert empty_faithfulness == 0.0

    def test_calculate_answer_relevancy(self):
        """Test 3: Test answer relevancy calculation."""
        relevancy = self.evaluator._calculate_answer_relevancy(
            self.sample_question, self.sample_answer
        )

        assert isinstance(relevancy, float)
        assert 0.0 <= relevancy <= 1.0

    def test_calculate_context_relevancy(self):
        """Test 4: Test context relevancy calculation."""
        context_relevancy = self.evaluator._calculate_context_relevancy(
            self.sample_question, self.sample_context
        )

        assert isinstance(context_relevancy, float)
        assert 0.0 <= context_relevancy <= 1.0

    def test_calculate_answer_correctness(self):
        """Test 5: Test answer correctness calculation."""
        correctness = self.evaluator._calculate_answer_correctness(
            self.sample_answer, self.sample_ground_truth
        )

        assert isinstance(correctness, float)
        assert 0.0 <= correctness <= 1.0

        # Test without ground truth
        no_gt_correctness = self.evaluator._calculate_answer_correctness(
            self.sample_answer, None
        )
        assert no_gt_correctness == 0.0

    def test_calculate_answer_similarity(self):
        """Test 6: Test answer similarity calculation."""
        similarity = self.evaluator._calculate_answer_similarity(
            self.sample_answer, self.sample_ground_truth
        )

        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0

    def test_calculate_context_recall(self):
        """Test 7: Test context recall calculation."""
        recall = self.evaluator._calculate_context_recall(
            self.sample_context, self.sample_ground_truth
        )

        assert isinstance(recall, float)
        assert 0.0 <= recall <= 1.0

    def test_calculate_context_precision(self):
        """Test 8: Test context precision calculation."""
        precision = self.evaluator._calculate_context_precision(
            self.sample_question, self.sample_context
        )

        assert isinstance(precision, float)
        assert 0.0 <= precision <= 1.0

    def test_evaluate_rag_response_full(self):
        """Test 9: Test complete RAG response evaluation."""
        metrics = self.evaluator.evaluate_rag_response(
            question=self.sample_question,
            answer=self.sample_answer,
            context=self.sample_context,
            ground_truth=self.sample_ground_truth
        )

        assert isinstance(metrics, RAGEvaluationMetrics)
        assert hasattr(metrics, 'faithfulness')
        assert hasattr(metrics, 'answer_relevancy')
        assert hasattr(metrics, 'context_relevancy')
        assert hasattr(metrics, 'answer_correctness')
        assert hasattr(metrics, 'answer_similarity')
        assert hasattr(metrics, 'context_recall')
        assert hasattr(metrics, 'context_precision')

        # All metrics should be floats between 0 and 1
        for attr_name in ['faithfulness', 'answer_relevancy', 'context_relevancy',
                         'answer_correctness', 'answer_similarity', 'context_recall', 'context_precision']:
            value = getattr(metrics, attr_name)
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

    def test_evaluate_with_thresholds(self):
        """Test 10: Test threshold evaluation."""
        metrics = self.evaluator.evaluate_rag_response(
            question=self.sample_question,
            answer=self.sample_answer,
            context=self.sample_context,
            ground_truth=self.sample_ground_truth
        )

        results = self.evaluator.evaluate_with_thresholds(metrics)

        assert isinstance(results, dict)
        assert len(results) >= 3  # At least faithfulness, answer_relevancy, context_relevancy

        for metric_name, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert result.metric_name == metric_name
            assert isinstance(result.score, float)
            assert isinstance(result.threshold, float)
            assert isinstance(result.passed, bool)
            assert isinstance(result.details, dict)
            assert isinstance(result.timestamp, datetime)

    def test_save_evaluation_results(self):
        """Test 11: Test saving evaluation results."""
        metrics = self.evaluator.evaluate_rag_response(
            question=self.sample_question,
            answer=self.sample_answer,
            context=self.sample_context,
            ground_truth=self.sample_ground_truth
        )

        results = self.evaluator.evaluate_with_thresholds(metrics)
        saved_file = self.evaluator.save_evaluation_results(results)

        assert saved_file.endswith('.json')
        assert Path(saved_file).exists()

        # Verify file contents
        with open(saved_file, 'r') as f:
            data = json.load(f)

        # Check that the saved data contains metric results directly (not nested under 'results')
        assert isinstance(data, dict)
        assert len(data) > 0

        # Check that at least some expected metrics are present
        expected_metrics = ['faithfulness', 'answer_relevancy', 'context_relevancy']
        found_metrics = [key for key in data.keys() if key in expected_metrics]
        assert len(found_metrics) > 0

        # Verify structure of saved metric data
        for metric_name, metric_data in data.items():
            assert 'metric_name' in metric_data
            assert 'score' in metric_data
            assert 'threshold' in metric_data
            assert 'passed' in metric_data
            assert 'details' in metric_data
            assert 'timestamp' in metric_data

    def test_generate_evaluation_report(self):
        """Test 12: Test evaluation report generation."""
        metrics = self.evaluator.evaluate_rag_response(
            question=self.sample_question,
            answer=self.sample_answer,
            context=self.sample_context,
            ground_truth=self.sample_ground_truth
        )

        results = self.evaluator.evaluate_with_thresholds(metrics)
        report = self.evaluator.generate_evaluation_report(results)

        assert isinstance(report, str)
        assert "RAG SYSTEM EVALUATION REPORT" in report
        assert "OVERALL PERFORMANCE" in report

    def test_get_evaluation_summary(self):
        """Test 13: Test evaluation summary generation."""
        # Add some evaluation results first
        metrics = self.evaluator.evaluate_rag_response(
            question=self.sample_question,
            answer=self.sample_answer,
            context=self.sample_context,
            ground_truth=self.sample_ground_truth
        )

        results = self.evaluator.evaluate_with_thresholds(metrics)

        summary = self.evaluator.get_evaluation_summary()

        assert isinstance(summary, dict)
        assert 'total_evaluations' in summary
        assert 'passed_evaluations' in summary
        assert 'pass_rate' in summary
        assert 'metric_summary' in summary

    def test_edge_cases(self):
        """Test 14: Test edge cases and error handling."""
        # Test with empty strings
        metrics = self.evaluator.evaluate_rag_response(
            question="",
            answer="",
            context=[],
            ground_truth=""
        )

        assert isinstance(metrics, RAGEvaluationMetrics)

        # Test with None values
        metrics_none = self.evaluator.evaluate_rag_response(
            question=None,
            answer=None,
            context=None,
            ground_truth=None
        )

        assert isinstance(metrics_none, RAGEvaluationMetrics)

    def test_threshold_customization(self):
        """Test 15: Test custom threshold configuration."""
        custom_thresholds = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.8,
            "context_relevancy": 0.8,
            "answer_correctness": 0.8,
            "answer_similarity": 0.8,
            "context_recall": 0.8,
            "context_precision": 0.8
        }

        evaluator = DeepEvalEvaluator()
        evaluator.thresholds = custom_thresholds

        metrics = evaluator.evaluate_rag_response(
            question=self.sample_question,
            answer=self.sample_answer,
            context=self.sample_context,
            ground_truth=self.sample_ground_truth
        )

        results = evaluator.evaluate_with_thresholds(metrics)

        # Check that custom thresholds are used
        assert results['faithfulness'].threshold == 0.8
        assert results['answer_relevancy'].threshold == 0.8


class TestRAGEvaluationMetrics:
    """Test cases for RAGEvaluationMetrics dataclass."""

    def test_metrics_dataclass(self):
        """Test 16: Test RAGEvaluationMetrics dataclass creation."""
        metrics = RAGEvaluationMetrics(
            faithfulness=0.8,
            answer_relevancy=0.7,
            context_relevancy=0.9,
            answer_correctness=0.6,
            answer_similarity=0.75,
            context_recall=0.85,
            context_precision=0.8
        )

        assert metrics.faithfulness == 0.8
        assert metrics.answer_relevancy == 0.7
        assert metrics.context_relevancy == 0.9
        assert metrics.answer_correctness == 0.6
        assert metrics.answer_similarity == 0.75
        assert metrics.context_recall == 0.85
        assert metrics.context_precision == 0.8


class TestEvaluationResult:
    """Test cases for EvaluationResult dataclass."""

    def test_evaluation_result_dataclass(self):
        """Test 17: Test EvaluationResult dataclass creation."""
        from datetime import datetime

        result = EvaluationResult(
            metric_name="faithfulness",
            score=0.8,
            threshold=0.7,
            passed=True,
            details={"description": "Test metric"},
            timestamp=datetime.now()
        )

        assert result.metric_name == "faithfulness"
        assert result.score == 0.8
        assert result.threshold == 0.7
        assert result.passed == True
        assert result.details["description"] == "Test metric"
        assert isinstance(result.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__])
