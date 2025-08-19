"""
Batch evaluation script for pure LLM (baseline)
Processes legal questions using OpenAI API directly without L-MARS enhancements
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lmars.evaluation import LegalAnswerEvaluator
from lmars.llm_judge import LLMJudgeEvaluator
from rate_limit_config import RateLimitHandler

load_dotenv()

# Configuration
EVAL_OPENAI_BASE_URL = os.getenv("EVAL_OPENAI_BASE_URL")
EVAL_OPENAI_API_KEY = os.getenv("EVAL_OPENAI_API_KEY")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))  # Reduced concurrent API calls to avoid rate limits
MODEL_NAME = os.getenv("EVAL_MODEL", "gpt-4o")

PROMPT_TEMPLATE = """You are a helpful legal assistant. Please provide a comprehensive answer to the following legal question.

Question: {question}

Please provide:
1. A clear and accurate answer based on current law
2. Key legal points and considerations
3. Relevant disclaimers about legal advice

Answer:"""


class BaseLLMEvaluator:
    """Evaluator for pure LLM responses without L-MARS enhancements."""
    
    def __init__(self, dataset_path: str = "eval/dataset/uncertain_legal_cases.json",
                 results_dir: str = "eval/results"):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path: Path to the evaluation dataset
            results_dir: Directory to save results
        """
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=EVAL_OPENAI_BASE_URL,
            api_key=EVAL_OPENAI_API_KEY,
        )
        
        # Initialize evaluators
        self.quantitative_evaluator = LegalAnswerEvaluator()
        self.qualitative_evaluator = LLMJudgeEvaluator()
        
        # Load dataset
        self.dataset = self.load_dataset()
        
        # Rate limit handler
        self.rate_limiter = RateLimitHandler(base_delay=3.0, max_retries=3)
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the evaluation dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def _get_default_judgment(self):
        """Return a default judgment when rate limited."""
        from types import SimpleNamespace
        return SimpleNamespace(
            factual_accuracy=SimpleNamespace(level="Medium", justification="Rate limited - skipped"),
            evidence_grounding=SimpleNamespace(level="Low", justification="No sources provided"),
            clarity_reasoning=SimpleNamespace(level="Medium", justification="Rate limited - skipped"),
            uncertainty_awareness=SimpleNamespace(level="Medium", justification="Rate limited - skipped"),
            overall_usefulness=SimpleNamespace(level="Medium", justification="Rate limited - skipped"),
            summary="LLM judge evaluation skipped due to rate limit",
            strengths=[],
            weaknesses=[]
        )
    
    def process_single_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single question through the LLM.
        
        Args:
            item: Dataset item with 'id' and 'question'
            
        Returns:
            Result dictionary with answer and evaluation
        """
        question_id = item['id']
        question = item['question']
        
        start_time = time.time()
        
        try:
            # Call OpenAI API with rate limit handling
            response = self.rate_limiter.execute_with_retry(
                self.client.chat.completions.create,
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a helpful legal assistant."},
                    {"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            answer = response.choices[0].message.content
            
            # Evaluate the answer
            # Quantitative evaluation
            quant_metrics = self.quantitative_evaluator.evaluate(
                answer_text=answer,
                sources=[],  # No sources for pure LLM
                jurisdiction_context=None
            )
            
            # Qualitative evaluation (with rate limit handling)
            qual_judgment = None
            if self.qualitative_evaluator is not None:
                try:
                    qual_judgment = self.rate_limiter.execute_with_retry(
                        self.qualitative_evaluator.evaluate,
                        question=question,
                        answer=answer,
                        sources=[]
                    )
                except Exception as e:
                    # If all retries failed, use default judgment
                    print(f"Warning: LLM judge evaluation failed after retries: {e}")
                    qual_judgment = self._get_default_judgment()
            
            # Build result with qualitative evaluation if available
            result = {
                'id': question_id,
                'question': question,
                'answer': answer,
                'model': MODEL_NAME,
                'processing_time': time.time() - start_time,
                'evaluation': {
                    'quantitative': {
                        'u_score': quant_metrics.u_score,
                        'hedging_score': quant_metrics.hedging_score,
                        'temporal_vagueness': quant_metrics.temporal_vagueness,
                        'citation_score': quant_metrics.citation_score,
                        'jurisdiction_score': quant_metrics.jurisdiction_score,
                        'decisiveness_score': quant_metrics.decisiveness_score
                    }
                },
                'status': 'success'
            }
            
            # Add qualitative evaluation if LLM judge is enabled and successful
            if self.qualitative_evaluator is not None and qual_judgment is not None:
                result['evaluation']['qualitative'] = {
                    'factual_accuracy': qual_judgment.factual_accuracy.level,
                    'evidence_grounding': qual_judgment.evidence_grounding.level,
                    'clarity_reasoning': qual_judgment.clarity_reasoning.level,
                    'uncertainty_awareness': qual_judgment.uncertainty_awareness.level,
                    'overall_usefulness': qual_judgment.overall_usefulness.level,
                    'summary': qual_judgment.summary
                }
            
        except Exception as e:
            result = {
                'id': question_id,
                'question': question,
                'answer': None,
                'model': MODEL_NAME,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'status': 'failed'
            }
        
        return result
    
    def run_batch_evaluation(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Run batch evaluation on the dataset.
        
        Args:
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            List of evaluation results
        """
        dataset = self.dataset[:max_samples] if max_samples else self.dataset
        results = []
        
        print(f"Starting batch evaluation of {len(dataset)} questions...")
        print(f"Using model: {MODEL_NAME}")
        print(f"Max workers: {MAX_WORKERS}")
        
        # Process with thread pool for concurrent API calls
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(self.process_single_question, item): item 
                for item in dataset
            }
            
            # Process completed tasks with progress bar
            with tqdm(total=len(dataset), desc="Processing") as pbar:
                for future in as_completed(future_to_item):
                    result = future.result()
                    results.append(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    if result['status'] == 'success':
                        pbar.set_postfix({'U-Score': f"{result['evaluation']['quantitative']['u_score']:.3f}"})
                    else:
                        pbar.set_postfix({'Status': 'Failed'})
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], run_name: str = None):
        """
        Save evaluation results to file.
        
        Args:
            results: List of evaluation results
            run_name: Optional name for this run
        """
        if not run_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"base_llm_{MODEL_NAME}_{timestamp}"
        
        output_file = self.results_dir / f"{run_name}.json"
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['status'] == 'success']
        
        summary = {
            'run_name': run_name,
            'model': MODEL_NAME,
            'timestamp': datetime.now().isoformat(),
            'total_questions': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'average_u_score': sum(r['evaluation']['quantitative']['u_score'] 
                                 for r in successful_results) / len(successful_results) if successful_results else 0,
            'average_processing_time': sum(r['processing_time'] 
                                          for r in successful_results) / len(successful_results) if successful_results else 0
        }
        
        # Save full results
        full_results = {
            'summary': summary,
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print(f"Summary:")
        print(f"  Total questions: {summary['total_questions']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Average U-Score: {summary['average_u_score']:.3f}")
        print(f"  Average processing time: {summary['average_processing_time']:.2f}s")
        
        return output_file


def main():
    """Main entry point for batch evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluation for pure LLM baseline")
    parser.add_argument('--dataset', default='eval/dataset/uncertain_legal_cases.json',
                       help='Path to evaluation dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--run-name', default=None,
                       help='Name for this evaluation run')
    parser.add_argument('--model', default=None,
                       help='Model to use (overrides EVAL_MODEL env var)')
    parser.add_argument('--no-judge', action='store_true',
                       help='Skip LLM judge evaluation (only do quantitative)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of concurrent workers (default: 2)')
    
    args = parser.parse_args()
    
    # Override model if specified
    if args.model:
        global MODEL_NAME
        MODEL_NAME = args.model
    
    # Override workers if specified
    if args.workers:
        global MAX_WORKERS
        MAX_WORKERS = args.workers
    
    # Run evaluation
    evaluator = BaseLLMEvaluator(dataset_path=args.dataset)
    
    # Optionally disable LLM judge to avoid rate limits
    if args.no_judge:
        evaluator.qualitative_evaluator = None
    
    results = evaluator.run_batch_evaluation(max_samples=args.max_samples)
    evaluator.save_results(results, run_name=args.run_name)


if __name__ == "__main__":
    main()