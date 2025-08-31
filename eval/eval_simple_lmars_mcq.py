"""
MCQ evaluation script for L-MARS Simple Mode
Processes multiple-choice questions using L-MARS pipeline with modified summary agent
"""
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from lmars.workflow import create_workflow
from lmars.evaluation import LegalAnswerEvaluator
from lmars.agents import SummaryAgent, SearchResult, FinalAnswer
from langchain_core.messages import HumanMessage
import logging

load_dotenv()

# Configuration
MODEL_NAME = os.getenv("EVAL_MODEL", "openai:gpt-4o")
ENABLE_OFFLINE_RAG = os.getenv("ENABLE_OFFLINE_RAG", "false").lower() == "true"
ENABLE_COURTLISTENER = os.getenv("ENABLE_COURTLISTENER", "false").lower() == "true"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MCQAnswer(BaseModel):
    """Multiple choice answer with reasoning"""
    selected_answer: str = Field(description="The selected answer (A, B, C, or D)")
    reasoning: str = Field(description="Brief reasoning for the selection")
    confidence: float = Field(description="Confidence level from 0 to 1")

class MCQFinalAnswer(FinalAnswer):
    """Extended FinalAnswer for MCQ"""
    mcq_answer: Optional[MCQAnswer] = None

class MCQSummaryAgent(SummaryAgent):
    """Modified summary agent that can handle MCQ questions"""
    
    def generate_mcq_answer(self, user_query: str, search_results: List[SearchResult], 
                           choices: Dict[str, str]) -> MCQFinalAnswer:
        """Generate answer for multiple choice question based on search results."""
        
        # First get the regular answer
        regular_answer = self.generate_final_answer(user_query, search_results)
        
        # Now select from MCQ choices
        structured_llm = self.llm.with_structured_output(MCQAnswer)
        
        results_content = "\n\n".join([
            f"From {r.source}:\n{r.content}"
            for r in search_results
        ])
        
        mcq_prompt = f"""
        Based on the search results, answer this multiple-choice question:
        
        Question: {user_query}
        
        Search Results:
        {results_content}
        
        Choose the BEST answer from these options:
        A: {choices['A']}
        B: {choices['B']}
        C: {choices['C']}
        D: {choices['D']}
        
        Instructions:
        1. Analyze the search results carefully
        2. Select the answer that is MOST supported by the evidence
        3. Your selected_answer must be exactly one of: A, B, C, or D
        4. Provide brief reasoning based on the search results
        5. Rate your confidence from 0 to 1
        """
        
        try:
            mcq_response = structured_llm.invoke([HumanMessage(content=mcq_prompt)])
            
            # Create extended answer
            final_answer = MCQFinalAnswer(
                answer=regular_answer.answer,
                sources=regular_answer.sources,
                key_points=regular_answer.key_points,
                disclaimers=regular_answer.disclaimers,
                mcq_answer=mcq_response
            )
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Error generating MCQ answer: {e}")
            # Fallback to regular answer
            return MCQFinalAnswer(
                answer=regular_answer.answer,
                sources=regular_answer.sources,
                key_points=regular_answer.key_points,
                disclaimers=regular_answer.disclaimers,
                mcq_answer=MCQAnswer(
                    selected_answer="ERROR",
                    reasoning=str(e),
                    confidence=0.0
                )
            )

class MCQEvaluator:
    """Evaluator for L-MARS MCQ Mode."""
    
    def __init__(self, dataset_path: str = "eval/qa_f1_rules.json",
                 results_dir: str = "eval/mcq_results"):
        """
        Initialize the MCQ evaluator.
        
        Args:
            dataset_path: Path to the MCQ dataset
            results_dir: Directory to save results
        """
        self.dataset_path = Path(dataset_path)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize quantitative evaluator only (no LLM judge for MCQ)
        self.quantitative_evaluator = LegalAnswerEvaluator()
        
        # Load dataset
        self.dataset = self.load_dataset()
        
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the MCQ dataset."""
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def process_single_mcq(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single MCQ through L-MARS pipeline.
        
        Args:
            item: Dataset item with 'id', 'mc_question', 'choices', and 'correct_answer'
            
        Returns:
            Result dictionary with answer and evaluation
        """
        question_id = item['id']
        question = item['mc_question']
        choices = item['choices']
        correct_answer = item['correct_answer']
        
        start_time = time.time()
        
        try:
            # Create workflow for this question
            workflow = create_workflow(
                mode="simple",
                llm_model=MODEL_NAME,
                enable_tracking=False,
                enable_offline_rag=ENABLE_OFFLINE_RAG,
                enable_courtlistener=ENABLE_COURTLISTENER
            )
            
            # Access the inner SimpleWorkflow and its summary agent
            simple_workflow = workflow.workflow
            original_summary_agent = simple_workflow.summary_agent
            
            # Replace the summary agent with MCQ version
            mcq_summary_agent = MCQSummaryAgent(original_summary_agent.llm)
            
            # Run the workflow with just the question (no choices shown during search)
            search_result = workflow.run(question)
            
            # Get search results
            search_results = search_result.get('search_results', [])
            
            # Generate MCQ answer using modified summary agent
            final_answer = mcq_summary_agent.generate_mcq_answer(
                question, 
                search_results, 
                choices
            )
            
            # Calculate accuracy
            is_correct = False
            selected_answer = "ERROR"
            
            if final_answer.mcq_answer:
                selected_answer = final_answer.mcq_answer.selected_answer.upper()
                is_correct = selected_answer == correct_answer.upper()
            
            processing_time = time.time() - start_time
            
            # Get quantitative evaluation (u-score)
            quantitative_eval = self.quantitative_evaluator.evaluate(final_answer.answer)
            
            # Convert EvaluationMetrics to dict if needed
            quantitative_dict = quantitative_eval.__dict__ if hasattr(quantitative_eval, '__dict__') else quantitative_eval
            
            return {
                'id': question_id,
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'model_answer': selected_answer,
                'is_correct': is_correct,
                'reasoning': final_answer.mcq_answer.reasoning if final_answer.mcq_answer else "",
                'confidence': final_answer.mcq_answer.confidence if final_answer.mcq_answer else 0.0,
                'full_answer': final_answer.answer,
                'sources': final_answer.sources,
                'key_points': final_answer.key_points,
                'search_sources': [
                    {
                        'source': sr.source,
                        'title': sr.title if hasattr(sr, 'title') else '',
                        'content_length': len(sr.content) if hasattr(sr, 'content') else 0
                    }
                    for sr in search_results
                ],
                'model': MODEL_NAME,
                'mode': 'simple_mcq',
                'processing_time': processing_time,
                'evaluation': {
                    'quantitative': quantitative_dict,
                    'accuracy': is_correct  # New accuracy metric
                },
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Error processing question {question_id}: {e}")
            return {
                'id': question_id,
                'question': question,
                'choices': choices,
                'correct_answer': correct_answer,
                'model_answer': 'ERROR',
                'is_correct': False,
                'reasoning': str(e),
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'evaluation': {
                    'accuracy': False
                },
                'status': 'failed',
                'error': str(e)
            }
    
    def run_evaluation(self, max_questions: Optional[int] = None) -> Dict[str, Any]:
        """
        Run evaluation on the entire dataset.
        
        Args:
            max_questions: Maximum number of questions to evaluate (None for all)
            
        Returns:
            Summary of evaluation results
        """
        questions_to_eval = self.dataset[:max_questions] if max_questions else self.dataset
        
        results = []
        correct_count = 0
        
        logger.info(f"Starting MCQ evaluation with {len(questions_to_eval)} questions")
        logger.info(f"Model: {MODEL_NAME}")
        logger.info(f"Mode: simple_mcq")
        
        for item in tqdm(questions_to_eval, desc="Processing MCQs"):
            result = self.process_single_mcq(item)
            results.append(result)
            
            if result['is_correct']:
                correct_count += 1
            
            # Log progress
            current_accuracy = (correct_count / len(results)) * 100
            logger.info(f"Q{result['id']}: {result['model_answer']} vs {result['correct_answer']} "
                       f"({'✓' if result['is_correct'] else '✗'}) | "
                       f"Accuracy: {current_accuracy:.1f}%")
        
        # Calculate summary statistics
        successful_results = [r for r in results if r['status'] == 'success']
        failed_results = [r for r in results if r['status'] == 'failed']
        
        accuracy = (correct_count / len(successful_results)) * 100 if successful_results else 0
        
        # Calculate average u-score
        u_scores = []
        for r in successful_results:
            if 'quantitative' in r['evaluation']:
                quant = r['evaluation']['quantitative']
                if isinstance(quant, dict) and 'u_score' in quant:
                    u_scores.append(quant['u_score'])
                elif hasattr(quant, 'u_score'):
                    u_scores.append(quant.u_score)
        avg_u_score = sum(u_scores) / len(u_scores) if u_scores else 0
        
        # Calculate average processing time
        processing_times = [r['processing_time'] for r in successful_results]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Extract dataset name from input file path
        dataset_name = os.path.splitext(os.path.basename(str(self.dataset_path)))[0]
        
        # Create summary
        summary = {
            'run_name': f"{dataset_name}_lmars_{MODEL_NAME.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'model': MODEL_NAME,
            'mode': 'simple_mcq',
            'timestamp': datetime.now().isoformat(),
            'dataset': str(self.dataset_path),
            'configuration': {
                'enable_offline_rag': ENABLE_OFFLINE_RAG,
                'enable_courtlistener': ENABLE_COURTLISTENER
            },
            'total_questions': len(questions_to_eval),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'average_u_score': avg_u_score,
            'average_processing_time': avg_processing_time,
            'answer_distribution': self._calculate_answer_distribution(results),
            'confidence_stats': self._calculate_confidence_stats(successful_results)
        }
        
        # Full results
        full_results = {
            'summary': summary,
            'results': results
        }
        
        # Save results with dataset name prefix
        output_filename = f"{dataset_name}_lmars_{MODEL_NAME.replace(':', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = self.results_dir / output_filename
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluation Complete!")
        logger.info(f"Total Questions: {summary['total_questions']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Accuracy: {summary['accuracy']:.2f}%")
        logger.info(f"Average U-Score: {summary['average_u_score']:.4f}")
        logger.info(f"Average Processing Time: {summary['average_processing_time']:.2f}s")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"{'='*60}\n")
        
        return full_results
    
    def _calculate_answer_distribution(self, results: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of selected answers."""
        distribution = {"A": 0, "B": 0, "C": 0, "D": 0, "ERROR": 0}
        for r in results:
            answer = r.get('model_answer', 'ERROR')
            if answer in distribution:
                distribution[answer] += 1
        return distribution
    
    def _calculate_confidence_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate confidence statistics."""
        confidences = [r.get('confidence', 0.0) for r in results]
        
        if not confidences:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}
        
        return {
            "mean": sum(confidences) / len(confidences),
            "min": min(confidences),
            "max": max(confidences)
        }

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate L-MARS on MCQ dataset')
    parser.add_argument('--dataset', type=str, 
                       default='eval/qa_f1_rules.json',
                       help='Path to MCQ dataset')
    parser.add_argument('--max-questions', type=int, 
                       default=None,
                       help='Maximum number of questions to evaluate')
    parser.add_argument('--model', type=str,
                       default='openai:gpt-4o',
                       help='Model to use for evaluation')
    
    args = parser.parse_args()
    
    # Set model from args
    os.environ['EVAL_MODEL'] = args.model
    
    # Run evaluation
    evaluator = MCQEvaluator(dataset_path=args.dataset)
    evaluator.run_evaluation(max_questions=args.max_questions)

if __name__ == "__main__":
    main()