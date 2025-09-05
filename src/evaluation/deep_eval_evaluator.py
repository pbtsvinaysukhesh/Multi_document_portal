
import os
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Data class for storing evaluation results."""
    metric_name: str
    score: float
    threshold: float
    passed: bool
    details: Dict[str, Any]
    timestamp: datetime

@dataclass
class RAGEvaluationMetrics:
    """Data class for RAG-specific evaluation metrics."""
    faithfulness: float
    answer_relevancy: float
    context_relevancy: float
    answer_correctness: float
    answer_similarity: float
    context_recall: float
    context_precision: float

class DeepEvalEvaluator:
    """DeepEval-based evaluator for RAG system performance."""

    def __init__(self, results_dir: str = "evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load environment variables for API keys
        api_keys = json.loads(os.getenv("API_KEYS", "{}"))
        self.groq_api_key = api_keys.get("GROQ_API_KEY")

        # Initialize metrics thresholds
        self.thresholds = {
            "faithfulness": 0.7,
            "answer_relevancy": 0.7,
            "context_relevancy": 0.7,
            "answer_correctness": 0.7,
            "answer_similarity": 0.7,
            "context_recall": 0.7,
            "context_precision": 0.7
        }

        self.evaluation_history: List[EvaluationResult] = []

        # Initialize in-memory cache for evaluation results
        self._evaluation_cache: Dict[str, RAGEvaluationMetrics] = {}
        self._threshold_cache: Dict[str, Dict[str, EvaluationResult]] = {}

    def evaluate_rag_response(self, question: str, answer: str, context: List[str],
                            ground_truth: Optional[str] = None) -> RAGEvaluationMetrics:
        """
        Evaluate a RAG response using multiple metrics with caching for performance.

        Args:
            question: The user's question
            answer: The generated answer
            context: List of retrieved context chunks
            ground_truth: Optional ground truth answer for comparison

        Returns:
            RAGEvaluationMetrics object with all evaluation scores
        """
        # Generate cache key based on inputs
        cache_key = self._generate_cache_key(question, answer, context, ground_truth)

        # Check cache first
        if cache_key in self._evaluation_cache:
            logger.info("Using cached evaluation results for performance optimization")
            return self._evaluation_cache[cache_key]

        try:
            # Calculate Faithfulness (how well the answer matches the context)
            faithfulness_score = self._calculate_faithfulness(answer, context)

            # Calculate Answer Relevancy (how relevant the answer is to the question)
            answer_relevancy_score = self._calculate_answer_relevancy(question, answer)

            # Calculate Context Relevancy (how relevant the retrieved context is)
            context_relevancy_score = self._calculate_context_relevancy(question, context)

            # Calculate Answer Correctness (if ground truth is available)
            answer_correctness_score = self._calculate_answer_correctness(answer, ground_truth) if ground_truth else 0.0

            # Calculate Answer Similarity (semantic similarity to ground truth)
            answer_similarity_score = self._calculate_answer_similarity(answer, ground_truth) if ground_truth else 0.0

            # Calculate Context Recall (how much of the relevant information was retrieved)
            context_recall_score = self._calculate_context_recall(context, ground_truth) if ground_truth else 0.0

            # Calculate Context Precision (how much of the retrieved information is relevant)
            context_precision_score = self._calculate_context_precision(question, context)

            metrics = RAGEvaluationMetrics(
                faithfulness=faithfulness_score,
                answer_relevancy=answer_relevancy_score,
                context_relevancy=context_relevancy_score,
                answer_correctness=answer_correctness_score,
                answer_similarity=answer_similarity_score,
                context_recall=context_recall_score,
                context_precision=context_precision_score
            )

            # Cache the results for future use
            self._evaluation_cache[cache_key] = metrics

            logger.info(f"RAG Evaluation completed: Faithfulness={faithfulness_score:.3f}, "
                       f"Answer Relevancy={answer_relevancy_score:.3f}")

            return metrics

        except Exception as e:
            logger.error(f"Error during RAG evaluation: {str(e)}")
            # Return default metrics on error
            default_metrics = RAGEvaluationMetrics(
                faithfulness=0.0, answer_relevancy=0.0, context_relevancy=0.0,
                answer_correctness=0.0, answer_similarity=0.0,
                context_recall=0.0, context_precision=0.0
            )
            # Cache default metrics to avoid repeated errors
            self._evaluation_cache[cache_key] = default_metrics
            return default_metrics

    def _calculate_faithfulness(self, answer: str, context: List[str]) -> float:
        """Calculate how faithful the answer is to the provided context."""
        try:
            # Simple implementation - check if answer contains context keywords
            context_text = ' '.join(context).lower()
            answer_words = set(answer.lower().split())
            context_words = set(context_text.split())

            # Calculate overlap
            overlap = len(answer_words.intersection(context_words))
            total_answer_words = len(answer_words)

            if total_answer_words == 0:
                return 0.0

            faithfulness = overlap / total_answer_words
            return min(faithfulness, 1.0)  # Cap at 1.0

        except Exception as e:
            logger.warning(f"Error calculating faithfulness: {str(e)}")
            return 0.0

    def _calculate_answer_relevancy(self, question: str, answer: str) -> float:
        """Calculate how relevant the answer is to the question."""
        try:
            question_words = set(question.lower().split())
            answer_words = set(answer.lower().split())

            # Calculate semantic overlap
            overlap = len(question_words.intersection(answer_words))
            total_question_words = len(question_words)

            if total_question_words == 0:
                return 0.0

            relevancy = overlap / total_question_words
            return min(relevancy, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating answer relevancy: {str(e)}")
            return 0.0

    def _calculate_context_relevancy(self, question: str, context: List[str]) -> float:
        """Calculate how relevant the retrieved context is to the question."""
        try:
            question_words = set(question.lower().split())
            context_text = ' '.join(context).lower()
            context_words = set(context_text.split())

            overlap = len(question_words.intersection(context_words))
            total_question_words = len(question_words)

            if total_question_words == 0:
                return 0.0

            relevancy = overlap / total_question_words
            return min(relevancy, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating context relevancy: {str(e)}")
            return 0.0

    def _calculate_answer_correctness(self, answer: str, ground_truth: str) -> float:
        """Calculate how correct the answer is compared to ground truth."""
        try:
            if not ground_truth:
                return 0.0

            # Simple word overlap calculation
            answer_words = set(answer.lower().split())
            truth_words = set(ground_truth.lower().split())

            overlap = len(answer_words.intersection(truth_words))
            total_truth_words = len(truth_words)

            if total_truth_words == 0:
                return 0.0

            correctness = overlap / total_truth_words
            return min(correctness, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating answer correctness: {str(e)}")
            return 0.0

    def _calculate_answer_similarity(self, answer: str, ground_truth: str) -> float:
        """Calculate semantic similarity between answer and ground truth."""
        try:
            if not ground_truth:
                return 0.0

            # Simple Jaccard similarity
            answer_words = set(answer.lower().split())
            truth_words = set(ground_truth.lower().split())

            intersection = len(answer_words.intersection(truth_words))
            union = len(answer_words.union(truth_words))

            if union == 0:
                return 0.0

            similarity = intersection / union
            return similarity

        except Exception as e:
            logger.warning(f"Error calculating answer similarity: {str(e)}")
            return 0.0

    def _calculate_context_recall(self, context: List[str], ground_truth: str) -> float:
        """Calculate how much relevant information was retrieved."""
        try:
            if not ground_truth:
                return 0.0

            context_text = ' '.join(context).lower()
            truth_words = set(ground_truth.lower().split())
            context_words = set(context_text.split())

            overlap = len(truth_words.intersection(context_words))
            total_truth_words = len(truth_words)

            if total_truth_words == 0:
                return 0.0

            recall = overlap / total_truth_words
            return min(recall, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating context recall: {str(e)}")
            return 0.0

    def _calculate_context_precision(self, question: str, context: List[str]) -> float:
        """Calculate how much of the retrieved information is relevant."""
        try:
            question_words = set(question.lower().split())
            context_text = ' '.join(context).lower()
            context_words = set(context_text.split())

            overlap = len(question_words.intersection(context_words))
            total_context_words = len(context_words)

            if total_context_words == 0:
                return 0.0

            precision = overlap / total_context_words
            return min(precision, 1.0)

        except Exception as e:
            logger.warning(f"Error calculating context precision: {str(e)}")
            return 0.0

    def evaluate_with_thresholds(self, metrics: RAGEvaluationMetrics) -> Dict[str, EvaluationResult]:
        """Evaluate metrics against predefined thresholds."""
        results = {}

        # Evaluate each metric
        results['faithfulness'] = EvaluationResult(
            metric_name='faithfulness',
            score=metrics.faithfulness,
            threshold=self.thresholds['faithfulness'],
            passed=metrics.faithfulness >= self.thresholds['faithfulness'],
            details={'description': 'How well the answer matches the context'},
            timestamp=datetime.now()
        )

        results['answer_relevancy'] = EvaluationResult(
            metric_name='answer_relevancy',
            score=metrics.answer_relevancy,
            threshold=self.thresholds['answer_relevancy'],
            passed=metrics.answer_relevancy >= self.thresholds['answer_relevancy'],
            details={'description': 'How relevant the answer is to the question'},
            timestamp=datetime.now()
        )

        results['context_relevancy'] = EvaluationResult(
            metric_name='context_relevancy',
            score=metrics.context_relevancy,
            threshold=self.thresholds['context_relevancy'],
            passed=metrics.context_relevancy >= self.thresholds['context_relevancy'],
            details={'description': 'How relevant the retrieved context is'},
            timestamp=datetime.now()
        )

        if metrics.answer_correctness > 0:
            results['answer_correctness'] = EvaluationResult(
                metric_name='answer_correctness',
                score=metrics.answer_correctness,
                threshold=self.thresholds['answer_correctness'],
                passed=metrics.answer_correctness >= self.thresholds['answer_correctness'],
                details={'description': 'How correct the answer is compared to ground truth'},
                timestamp=datetime.now()
            )

        if metrics.answer_similarity > 0:
            results['answer_similarity'] = EvaluationResult(
                metric_name='answer_similarity',
                score=metrics.answer_similarity,
                threshold=self.thresholds['answer_similarity'],
                passed=metrics.answer_similarity >= self.thresholds['answer_similarity'],
                details={'description': 'Semantic similarity to ground truth'},
                timestamp=datetime.now()
            )

        if metrics.context_recall > 0:
            results['context_recall'] = EvaluationResult(
                metric_name='context_recall',
                score=metrics.context_recall,
                threshold=self.thresholds['context_recall'],
                passed=metrics.context_recall >= self.thresholds['context_recall'],
                details={'description': 'How much relevant information was retrieved'},
                timestamp=datetime.now()
            )

        if metrics.context_precision > 0:
            results['context_precision'] = EvaluationResult(
                metric_name='context_precision',
                score=metrics.context_precision,
                threshold=self.thresholds['context_precision'],
                passed=metrics.context_precision >= self.thresholds['context_precision'],
                details={'description': 'How much retrieved information is relevant'},
                timestamp=datetime.now()
            )

        # Store results in history
        self.evaluation_history.extend(results.values())

        return results

    def save_evaluation_results(self, results: Dict[str, EvaluationResult],
                              filename: Optional[str] = None) -> str:
        """Save evaluation results to a JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"

        filepath = self.results_dir / filename

        # Convert results to serializable format
        serializable_results = {}
        for key, result in results.items():
            serializable_results[key] = {
                'metric_name': result.metric_name,
                'score': result.score,
                'threshold': result.threshold,
                'passed': result.passed,
                'details': result.details,
                'timestamp': result.timestamp.isoformat()
            }

        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Evaluation results saved to {filepath}")
        return str(filepath)

    def generate_evaluation_report(self, results: Dict[str, EvaluationResult]) -> str:
        """Generate a human-readable evaluation report."""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("RAG SYSTEM EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        passed_count = sum(1 for result in results.values() if result.passed)
        total_count = len(results)

        report_lines.append(f"OVERALL PERFORMANCE: {passed_count}/{total_count} metrics passed")
        report_lines.append("")

        for metric_name, result in results.items():
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            report_lines.append(f"{metric_name.upper()}: {result.score:.3f} (threshold: {result.threshold}) - {status}")
            report_lines.append(f"  Description: {result.details.get('description', 'N/A')}")
            report_lines.append("")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a summary of all evaluation history."""
        if not self.evaluation_history:
            return {"message": "No evaluation history available"}

        total_evaluations = len(self.evaluation_history)
        passed_evaluations = sum(1 for result in self.evaluation_history if result.passed)

        # Group by metric type
        metric_summary = {}
        for result in self.evaluation_history:
            if result.metric_name not in metric_summary:
                metric_summary[result.metric_name] = {
                    'count': 0,
                    'passed': 0,
                    'avg_score': 0.0,
                    'scores': []
                }

            metric_summary[result.metric_name]['count'] += 1
            metric_summary[result.metric_name]['scores'].append(result.score)
            if result.passed:
                metric_summary[result.metric_name]['passed'] += 1

        # Calculate averages
        for metric_data in metric_summary.values():
            metric_data['avg_score'] = sum(metric_data['scores']) / len(metric_data['scores'])

        return {
            'total_evaluations': total_evaluations,
            'passed_evaluations': passed_evaluations,
            'pass_rate': passed_evaluations / total_evaluations if total_evaluations > 0 else 0,
            'metric_summary': metric_summary
        }

    def _generate_cache_key(self, question: str, answer: str, context: List[str],
                           ground_truth: Optional[str] = None) -> str:
        """Generate a unique cache key based on evaluation inputs."""
        # Create a string representation of all inputs
        cache_string = f"{question}|{answer}|{'|'.join(context)}|{ground_truth or ''}"

        # Generate SHA256 hash for consistent cache key
        return hashlib.sha256(cache_string.encode('utf-8')).hexdigest()

    def clear_cache(self) -> None:
        """Clear all cached evaluation results."""
        self._evaluation_cache.clear()
        self._threshold_cache.clear()
        logger.info("Evaluation cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the current cache state."""
        return {
            'evaluation_cache_size': len(self._evaluation_cache),
            'threshold_cache_size': len(self._threshold_cache)
        }
