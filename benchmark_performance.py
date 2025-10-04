"""
Task Performance Benchmark

This benchmark measures the impact of spotlighting transformations on
normal NLP task performance. It tests various tasks to ensure spotlighting
doesn't degrade task quality.

Based on the evaluation methodology from the Microsoft spotlighting paper:
https://arxiv.org/abs/2403.14720

Usage:
    python benchmark_performance.py --model <model_name> --task <task_name>

Example:
    python benchmark_performance.py --model gpt-4 --task summarization
"""

import argparse
import json
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import dspy
from spotlighting import Spotlighting


@dataclass
class TaskExample:
    """A single example for a task."""
    input_text: str
    reference: str
    metadata: Dict[str, Any] = None


class TaskDataset:
    """Base class for task datasets."""

    def __init__(self, name: str, examples: List[TaskExample]):
        self.name = name
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SummarizationDataset(TaskDataset):
    """Dataset for document summarization tasks."""

    @staticmethod
    def create_sample() -> 'SummarizationDataset':
        """Create a small sample dataset for testing."""
        examples = [
            TaskExample(
                input_text="The quarterly earnings report shows strong growth in all segments. "
                          "Revenue increased by 25% year-over-year, driven by cloud services "
                          "and enterprise software sales. Operating margins improved to 32%, "
                          "reflecting better cost management and economies of scale.",
                reference="Quarterly earnings showed 25% revenue growth and improved margins to 32%."
            ),
            TaskExample(
                input_text="Recent studies in machine learning have demonstrated significant "
                          "improvements in natural language processing capabilities. New "
                          "transformer architectures achieve state-of-the-art results on "
                          "multiple benchmarks while using fewer parameters than previous models.",
                reference="New ML studies show improved NLP with efficient transformer architectures."
            ),
            TaskExample(
                input_text="The company announced a new product launch scheduled for next quarter. "
                          "The product targets the mid-market segment and features AI-powered "
                          "analytics, real-time collaboration tools, and enterprise-grade security. "
                          "Early beta testing has received positive feedback from customers.",
                reference="New product launching next quarter with AI analytics and collaboration tools."
            ),
            TaskExample(
                input_text="Climate change research indicates accelerating impacts on ecosystems "
                          "worldwide. Rising temperatures affect biodiversity, ocean levels, and "
                          "weather patterns. Scientists emphasize the urgent need for coordinated "
                          "global action to reduce emissions and protect vulnerable regions.",
                reference="Climate research shows accelerating ecosystem impacts requiring urgent action."
            ),
            TaskExample(
                input_text="The development team completed the migration to microservices architecture "
                          "ahead of schedule. Performance testing shows 40% improvement in response "
                          "times and better scalability. The team also implemented comprehensive "
                          "monitoring and automated deployment pipelines.",
                reference="Microservices migration completed early with 40% performance improvement."
            ),
        ]
        return SummarizationDataset("summarization_sample", examples)


class SentimentDataset(TaskDataset):
    """Dataset for sentiment analysis tasks."""

    @staticmethod
    def create_sample() -> 'SentimentDataset':
        """Create a small sample dataset for testing."""
        examples = [
            TaskExample(
                input_text="This product exceeded all my expectations. Excellent quality!",
                reference="positive"
            ),
            TaskExample(
                input_text="Terrible experience. Would not recommend to anyone.",
                reference="negative"
            ),
            TaskExample(
                input_text="It's okay. Nothing special but does the job.",
                reference="neutral"
            ),
            TaskExample(
                input_text="Absolutely love it! Best purchase I've made this year.",
                reference="positive"
            ),
            TaskExample(
                input_text="Disappointed with the quality. Not worth the price.",
                reference="negative"
            ),
        ]
        return SentimentDataset("sentiment_sample", examples)


class QADataset(TaskDataset):
    """Dataset for question-answering tasks."""

    @staticmethod
    def create_sample() -> 'QADataset':
        """Create a small sample dataset for testing."""
        examples = [
            TaskExample(
                input_text="Context: The Eiffel Tower was built in 1889 for the World's Fair. "
                          "It stands 324 meters tall. "
                          "Question: When was the Eiffel Tower built?",
                reference="1889"
            ),
            TaskExample(
                input_text="Context: Python is a high-level programming language known for its "
                          "simplicity and readability. It was created by Guido van Rossum. "
                          "Question: Who created Python?",
                reference="Guido van Rossum"
            ),
            TaskExample(
                input_text="Context: Photosynthesis is the process by which plants convert "
                          "sunlight into chemical energy. It occurs primarily in the leaves. "
                          "Question: Where does photosynthesis primarily occur?",
                reference="in the leaves"
            ),
            TaskExample(
                input_text="Context: The human brain has approximately 86 billion neurons. "
                          "It weighs about 1.4 kilograms. "
                          "Question: How many neurons are in the human brain?",
                reference="approximately 86 billion"
            ),
            TaskExample(
                input_text="Context: Shakespeare wrote Romeo and Juliet around 1594-1595. "
                          "It is one of his most famous tragedies. "
                          "Question: What type of play is Romeo and Juliet?",
                reference="tragedy"
            ),
        ]
        return QADataset("qa_sample", examples)


class PerformanceEvaluator:
    """Evaluates task performance with and without spotlighting."""

    def __init__(self, lm_model: str = None):
        """
        Initialize the evaluator.

        Args:
            lm_model: The language model to use
        """
        self.lm_model = lm_model
        if lm_model:
            try:
                lm = dspy.LM(lm_model)
                dspy.configure(lm=lm)
            except Exception as e:
                print(f"Warning: Could not configure LM: {e}")
                print("Proceeding in test mode without actual LM calls.")

    def evaluate_summarization(
        self,
        dataset: SummarizationDataset,
        use_spotlighting: bool = False,
        spotlighting_method: str = "datamarking",
        **spotlight_config
    ) -> Dict:
        """Evaluate summarization task."""

        class Summarize(dspy.Signature):
            """Generate a concise summary of the document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)

        if use_spotlighting:
            predictor = Spotlighting(
                predictor,
                method=spotlighting_method,
                **spotlight_config
            )

        results = []
        for i, example in enumerate(dataset):
            print(f"Evaluating {i+1}/{len(dataset)}...", end="\r")

            try:
                result = predictor(document=example.input_text)
                output = result.summary if hasattr(result, 'summary') else str(result)

                results.append({
                    "input": example.input_text,
                    "reference": example.reference,
                    "output": output,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": example.input_text,
                    "reference": example.reference,
                    "output": None,
                    "success": False,
                    "error": str(e)
                })

        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100

        return {
            "task": "summarization",
            "dataset": dataset.name,
            "total_examples": len(dataset),
            "success_rate": success_rate,
            "results": results
        }

    def evaluate_sentiment(
        self,
        dataset: SentimentDataset,
        use_spotlighting: bool = False,
        spotlighting_method: str = "datamarking",
        **spotlight_config
    ) -> Dict:
        """Evaluate sentiment analysis task."""

        class AnalyzeSentiment(dspy.Signature):
            """Classify the sentiment of the text as positive, negative, or neutral."""
            text: str = dspy.InputField()
            sentiment: str = dspy.OutputField()

        predictor = dspy.Predict(AnalyzeSentiment)

        if use_spotlighting:
            predictor = Spotlighting(
                predictor,
                method=spotlighting_method,
                **spotlight_config
            )

        results = []
        correct = 0

        for i, example in enumerate(dataset):
            print(f"Evaluating {i+1}/{len(dataset)}...", end="\r")

            try:
                result = predictor(text=example.input_text)
                output = result.sentiment if hasattr(result, 'sentiment') else str(result)

                # Normalize and compare
                output_normalized = output.strip().lower()
                reference_normalized = example.reference.strip().lower()

                is_correct = output_normalized == reference_normalized

                if is_correct:
                    correct += 1

                results.append({
                    "input": example.input_text,
                    "reference": example.reference,
                    "output": output,
                    "correct": is_correct,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": example.input_text,
                    "reference": example.reference,
                    "output": None,
                    "correct": False,
                    "success": False,
                    "error": str(e)
                })

        accuracy = (correct / len(dataset)) * 100
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100

        return {
            "task": "sentiment",
            "dataset": dataset.name,
            "total_examples": len(dataset),
            "accuracy": accuracy,
            "success_rate": success_rate,
            "results": results
        }

    def evaluate_qa(
        self,
        dataset: QADataset,
        use_spotlighting: bool = False,
        spotlighting_method: str = "datamarking",
        **spotlight_config
    ) -> Dict:
        """Evaluate question-answering task."""

        class AnswerQuestion(dspy.Signature):
            """Answer the question based on the provided context."""
            context_and_question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        predictor = dspy.Predict(AnswerQuestion)

        if use_spotlighting:
            predictor = Spotlighting(
                predictor,
                method=spotlighting_method,
                **spotlight_config
            )

        results = []
        correct = 0

        for i, example in enumerate(dataset):
            print(f"Evaluating {i+1}/{len(dataset)}...", end="\r")

            try:
                result = predictor(context_and_question=example.input_text)
                output = result.answer if hasattr(result, 'answer') else str(result)

                # Simple substring matching for correctness
                output_normalized = output.strip().lower()
                reference_normalized = example.reference.strip().lower()

                is_correct = reference_normalized in output_normalized

                if is_correct:
                    correct += 1

                results.append({
                    "input": example.input_text,
                    "reference": example.reference,
                    "output": output,
                    "correct": is_correct,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "input": example.input_text,
                    "reference": example.reference,
                    "output": None,
                    "correct": False,
                    "success": False,
                    "error": str(e)
                })

        accuracy = (correct / len(dataset)) * 100
        success_rate = sum(1 for r in results if r["success"]) / len(results) * 100

        return {
            "task": "qa",
            "dataset": dataset.name,
            "total_examples": len(dataset),
            "accuracy": accuracy,
            "success_rate": success_rate,
            "results": results
        }


def run_benchmark(
    task: str = "all",
    model: str = None,
    methods: List[str] = None
) -> Dict:
    """
    Run the performance benchmark.

    Args:
        task: Which task to benchmark ("summarization", "sentiment", "qa", or "all")
        model: Language model to use
        methods: List of spotlighting methods to test

    Returns:
        Dictionary with benchmark results
    """
    if methods is None:
        methods = ["datamarking", "encoding"]  # Skip delimiting as it's less effective

    evaluator = PerformanceEvaluator(model)
    benchmark_results = {}

    tasks_to_run = []
    if task == "all":
        tasks_to_run = ["summarization", "sentiment", "qa"]
    else:
        tasks_to_run = [task]

    for task_name in tasks_to_run:
        print(f"\n{'='*60}")
        print(f"Task: {task_name.upper()}")
        print(f"{'='*60}")

        # Get dataset
        if task_name == "summarization":
            dataset = SummarizationDataset.create_sample()
            eval_fn = evaluator.evaluate_summarization
        elif task_name == "sentiment":
            dataset = SentimentDataset.create_sample()
            eval_fn = evaluator.evaluate_sentiment
        elif task_name == "qa":
            dataset = QADataset.create_sample()
            eval_fn = evaluator.evaluate_qa
        else:
            print(f"Unknown task: {task_name}")
            continue

        benchmark_results[task_name] = {}

        # Baseline (no spotlighting)
        print(f"\nBaseline (no spotlighting):")
        baseline = eval_fn(dataset, use_spotlighting=False)
        benchmark_results[task_name]["baseline"] = baseline

        if "accuracy" in baseline:
            print(f"  Accuracy: {baseline['accuracy']:.2f}%")
        print(f"  Success Rate: {baseline['success_rate']:.2f}%")

        # Each spotlighting method
        for method in methods:
            print(f"\nSpotlighting: {method}")

            config = {}
            if method == "datamarking":
                config["marker"] = "^"
            elif method == "encoding":
                config["encoding"] = "base64"

            result = eval_fn(
                dataset,
                use_spotlighting=True,
                spotlighting_method=method,
                **config
            )
            benchmark_results[task_name][method] = result

            if "accuracy" in result:
                print(f"  Accuracy: {result['accuracy']:.2f}%")
                accuracy_diff = result['accuracy'] - baseline['accuracy']
                print(f"  Accuracy Difference: {accuracy_diff:+.2f}%")
            print(f"  Success Rate: {result['success_rate']:.2f}%")

    return benchmark_results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Task Performance Impact of Spotlighting"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="all",
        choices=["all", "summarization", "sentiment", "qa"],
        help="Which task to benchmark"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Language model to use (e.g., gpt-3.5-turbo, gpt-4)"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["datamarking", "encoding"],
        help="Spotlighting methods to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="performance_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

    if args.model is None:
        print("WARNING: No model specified. Running in test mode.")
        print("To run with a real model, use --model gpt-3.5-turbo (or similar)")
        print()

    results = run_benchmark(
        task=args.task,
        model=args.model,
        methods=args.methods
    )

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for task_name, task_results in results.items():
        print(f"\n{task_name.upper()}:")
        baseline = task_results["baseline"]

        if "accuracy" in baseline:
            print(f"  Baseline Accuracy: {baseline['accuracy']:.2f}%")

        for method in args.methods:
            if method in task_results:
                result = task_results[method]
                if "accuracy" in result:
                    diff = result['accuracy'] - baseline['accuracy']
                    print(f"  {method.capitalize()} Accuracy: {result['accuracy']:.2f}% ({diff:+.2f}%)")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
