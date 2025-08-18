"""
Batch evaluation script for L-MARS Multi-Turn Mode
Processes legal questions with iterative refinement and judge agent
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from lmars.workflow import create_workflow
from lmars.evaluation import LegalAnswerEvaluator
from lmars.llm_judge import LLMJudgeEvaluator

load_dotenv()

# Configuration
MODEL_NAME = os.getenv("EVAL_MODEL", "openai:gpt-4o")
JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", None)  # Use same as main model if not specified
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
ENABLE_OFFLINE_RAG = os.getenv("ENABLE_OFFLINE_RAG", "false").lower() == "true"
ENABLE_COURTLISTENER = os.getenv("ENABLE_COURTLISTENER", "false").lower() == "true"

# Default follow-up responses for automated evaluation
DEFAULT_FOLLOWUP_RESPONSES = {
    "jurisdiction": "United States federal law",
    "timeline": "Current as of 2024",
    "context": "General legal guidance needed",
    "purpose": "Educational research"
}


class MultiTurnModeEvaluator:
    """Evaluator for L-MARS Multi-Turn Mode."""
    
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
        
        # Initialize evaluators
        self.quantitative_evaluator = LegalAnswerEvaluator()
        self.qualitative_evaluator = LLMJudgeEvaluator()
        
        # Load dataset
        self.dataset = self.load_dataset()
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the evaluation dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def generate_followup_responses(self, follow_up_questions: List) -> Dict[str, str]:
        """
        Generate automated responses to follow-up questions.
        
        Args:
            follow_up_questions: List of follow-up questions from the system
            
        Returns:
            Dictionary of responses
        """
        responses = {}
        
        for i, q in enumerate(follow_up_questions, 1):
            question_text = q.question.lower() if hasattr(q, 'question') else str(q).lower()
            
            # Match common patterns and provide appropriate responses
            if 'jurisdiction' in question_text or 'location' in question_text or 'state' in question_text:
                responses[f"question_{i}"] = DEFAULT_FOLLOWUP_RESPONSES["jurisdiction"]
            elif 'timeline' in question_text or 'when' in question_text or 'date' in question_text:
                responses[f"question_{i}"] = DEFAULT_FOLLOWUP_RESPONSES["timeline"]
            elif 'context' in question_text or 'situation' in question_text or 'circumstance' in question_text:
                responses[f"question_{i}"] = DEFAULT_FOLLOWUP_RESPONSES["context"]
            elif 'purpose' in question_text or 'why' in question_text or 'help' in question_text:
                responses[f"question_{i}"] = DEFAULT_FOLLOWUP_RESPONSES["purpose"]
            else:
                # Generic response for unmatched questions
                responses[f"question_{i}"] = "General information needed for research purposes"
        
        return responses
    
    def process_single_question(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single question through L-MARS Multi-Turn Mode.
        
        Args:
            item: Dataset item with 'id' and 'question'
            
        Returns:
            Result dictionary with answer and evaluation
        """
        question_id = item['id']
        question = item['question']
        
        start_time = time.time()
        iterations_info = []
        
        try:
            # Create workflow for this question
            workflow = create_workflow(
                mode="multi_turn",
                llm_model=MODEL_NAME,
                judge_model=JUDGE_MODEL or MODEL_NAME,
                max_iterations=MAX_ITERATIONS,
                enable_tracking=False,
                enable_offline_rag=ENABLE_OFFLINE_RAG,
                enable_courtlistener=ENABLE_COURTLISTENER
            )
            
            # Initial run
            result = workflow.run(question)
            
            # Handle follow-up questions if needed
            if result.get("needs_user_input") and result.get("follow_up_questions"):
                follow_up_questions = result["follow_up_questions"]
                
                # Generate automated responses
                user_responses = self.generate_followup_responses(follow_up_questions)
                
                # Store follow-up info
                iterations_info.append({
                    "type": "follow_up_questions",
                    "questions": [q.question if hasattr(q, 'question') else str(q) 
                                for q in follow_up_questions],
                    "responses": user_responses
                })
                
                # Continue with responses
                result = workflow.run(question, user_responses)
            
            # Extract answer and search results
            final_answer = result.get('final_answer')
            search_results = result.get('search_results', [])
            iterations = result.get('iterations', 1)
            
            if final_answer:
                answer_text = final_answer.answer
                sources = final_answer.sources
                
                # Extract search sources info
                search_sources = []
                for sr in search_results:
                    if hasattr(sr, 'source'):
                        search_sources.append({
                            'source': sr.source,
                            'title': sr.title if hasattr(sr, 'title') else '',
                            'content_length': len(sr.content) if hasattr(sr, 'content') else 0
                        })
                
                # Evaluate the answer
                # Quantitative evaluation
                quant_metrics = self.quantitative_evaluator.evaluate(
                    answer_text=answer_text,
                    sources=sources,
                    jurisdiction_context=None
                )
                
                # Qualitative evaluation
                qual_judgment = self.qualitative_evaluator.evaluate(
                    question=question,
                    answer=answer_text,
                    sources=sources
                )
                
                result_dict = {
                    'id': question_id,
                    'question': question,
                    'answer': answer_text,
                    'sources': sources,
                    'search_sources': search_sources,
                    'key_points': final_answer.key_points,
                    'disclaimers': final_answer.disclaimers,
                    'model': MODEL_NAME,
                    'judge_model': JUDGE_MODEL or MODEL_NAME,
                    'mode': 'multi_turn',
                    'iterations': iterations,
                    'iterations_info': iterations_info,
                    'processing_time': time.time() - start_time,
                    'evaluation': {
                        'quantitative': {
                            'u_score': quant_metrics.u_score,
                            'hedging_score': quant_metrics.hedging_score,
                            'temporal_vagueness': quant_metrics.temporal_vagueness,
                            'citation_score': quant_metrics.citation_score,
                            'jurisdiction_score': quant_metrics.jurisdiction_score,
                            'decisiveness_score': quant_metrics.decisiveness_score
                        },
                        'qualitative': {
                            'factual_accuracy': qual_judgment.factual_accuracy.level,
                            'evidence_grounding': qual_judgment.evidence_grounding.level,
                            'clarity_reasoning': qual_judgment.clarity_reasoning.level,
                            'uncertainty_awareness': qual_judgment.uncertainty_awareness.level,
                            'overall_usefulness': qual_judgment.overall_usefulness.level,
                            'summary': qual_judgment.summary
                        }
                    },
                    'status': 'success'
                }
            else:
                result_dict = {
                    'id': question_id,
                    'question': question,
                    'answer': None,
                    'model': MODEL_NAME,
                    'judge_model': JUDGE_MODEL or MODEL_NAME,
                    'mode': 'multi_turn',
                    'iterations': 0,
                    'processing_time': time.time() - start_time,
                    'error': 'No final answer generated',
                    'status': 'failed'
                }
                
        except Exception as e:
            result_dict = {
                'id': question_id,
                'question': question,
                'answer': None,
                'model': MODEL_NAME,
                'judge_model': JUDGE_MODEL or MODEL_NAME,
                'mode': 'multi_turn',
                'iterations': 0,
                'processing_time': time.time() - start_time,
                'error': str(e),
                'status': 'failed'
            }
        
        return result_dict
    
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
        
        print(f"Starting L-MARS Multi-Turn Mode evaluation of {len(dataset)} questions...")
        print(f"Using model: {MODEL_NAME}")
        print(f"Judge model: {JUDGE_MODEL or MODEL_NAME}")
        print(f"Max iterations: {MAX_ITERATIONS}")
        print(f"Offline RAG: {ENABLE_OFFLINE_RAG}")
        print(f"CourtListener: {ENABLE_COURTLISTENER}")
        
        # Process questions sequentially
        for item in tqdm(dataset, desc="Processing"):
            result = self.process_single_question(item)
            results.append(result)
            
            # Show current status
            if result['status'] == 'success':
                tqdm.write(f"  ✓ ID {result['id']}: U-Score = {result['evaluation']['quantitative']['u_score']:.3f}, Iterations = {result['iterations']}")
            else:
                tqdm.write(f"  ✗ ID {result['id']}: Failed - {result.get('error', 'Unknown error')}")
        
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
            run_name = f"multi_turn_{MODEL_NAME.replace(':', '_')}_{timestamp}"
        
        output_file = self.results_dir / f"{run_name}.json"
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['status'] == 'success']
        
        # Calculate average scores
        avg_u_score = sum(r['evaluation']['quantitative']['u_score'] 
                         for r in successful_results) / len(successful_results) if successful_results else 0
        
        avg_iterations = sum(r['iterations'] 
                           for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Count qualitative levels
        qual_levels = {
            'factual_accuracy': {'High': 0, 'Medium': 0, 'Low': 0},
            'evidence_grounding': {'High': 0, 'Medium': 0, 'Low': 0},
            'clarity_reasoning': {'High': 0, 'Medium': 0, 'Low': 0},
            'uncertainty_awareness': {'High': 0, 'Medium': 0, 'Low': 0},
            'overall_usefulness': {'High': 0, 'Medium': 0, 'Low': 0}
        }
        
        for r in successful_results:
            for metric in qual_levels:
                level = r['evaluation']['qualitative'][metric]
                if level in qual_levels[metric]:
                    qual_levels[metric][level] += 1
        
        summary = {
            'run_name': run_name,
            'model': MODEL_NAME,
            'judge_model': JUDGE_MODEL or MODEL_NAME,
            'mode': 'multi_turn',
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'max_iterations': MAX_ITERATIONS,
                'enable_offline_rag': ENABLE_OFFLINE_RAG,
                'enable_courtlistener': ENABLE_COURTLISTENER
            },
            'total_questions': len(results),
            'successful': len(successful_results),
            'failed': len(results) - len(successful_results),
            'average_u_score': avg_u_score,
            'average_iterations': avg_iterations,
            'average_processing_time': sum(r['processing_time'] 
                                          for r in successful_results) / len(successful_results) if successful_results else 0,
            'qualitative_distribution': qual_levels
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
        print(f"  Average iterations: {summary['average_iterations']:.1f}")
        print(f"  Average processing time: {summary['average_processing_time']:.2f}s")
        print(f"  Overall Usefulness Distribution:")
        print(f"    High: {qual_levels['overall_usefulness']['High']}")
        print(f"    Medium: {qual_levels['overall_usefulness']['Medium']}")
        print(f"    Low: {qual_levels['overall_usefulness']['Low']}")
        
        return output_file


def main():
    """Main entry point for batch evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch evaluation for L-MARS Multi-Turn Mode")
    parser.add_argument('--dataset', default='eval/dataset/uncertain_legal_cases.json',
                       help='Path to evaluation dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--run-name', default=None,
                       help='Name for this evaluation run')
    parser.add_argument('--model', default=None,
                       help='Model to use (overrides EVAL_MODEL env var)')
    parser.add_argument('--judge-model', default=None,
                       help='Judge model to use (overrides EVAL_JUDGE_MODEL env var)')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum iterations (overrides MAX_ITERATIONS env var)')
    parser.add_argument('--enable-offline-rag', action='store_true',
                       help='Enable offline RAG search')
    parser.add_argument('--enable-courtlistener', action='store_true',
                       help='Enable CourtListener search')
    
    args = parser.parse_args()
    
    # Override settings if specified
    if args.model:
        global MODEL_NAME
        MODEL_NAME = args.model
    if args.judge_model:
        global JUDGE_MODEL
        JUDGE_MODEL = args.judge_model
    if args.max_iterations:
        global MAX_ITERATIONS
        MAX_ITERATIONS = args.max_iterations
    if args.enable_offline_rag:
        global ENABLE_OFFLINE_RAG
        ENABLE_OFFLINE_RAG = True
    if args.enable_courtlistener:
        global ENABLE_COURTLISTENER
        ENABLE_COURTLISTENER = True
    
    # Run evaluation
    evaluator = MultiTurnModeEvaluator(dataset_path=args.dataset)
    results = evaluator.run_batch_evaluation(max_samples=args.max_samples)
    evaluator.save_results(results, run_name=args.run_name)


if __name__ == "__main__":
    main()