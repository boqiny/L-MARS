"""
Simple MCQ evaluation pipeline for OpenAI models
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
from openai import OpenAI
from dotenv import load_dotenv
import time
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class MCQAnswer(BaseModel):
    answer: str
    reasoning: str

def get_client():
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

def evaluate_question(question_data, model="gpt-5"):
    """Evaluate a single MCQ question"""
    try:
        client = get_client()
        start = time.time()
        
        prompt = f"""Answer this multiple-choice question:

{question_data['mc_question']}

A: {question_data['choices']['A']}
B: {question_data['choices']['B']}
C: {question_data['choices']['C']}
D: {question_data['choices']['D']}

Choose the best answer (A, B, C, or D) and explain briefly."""
        
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": "You are answering a legal multiple-choice question. Answer with just the letter (A, B, C, or D) and brief reasoning."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=300,
            response_format=MCQAnswer
        )
        
        model_answer = response.choices[0].message.parsed.answer.upper()
        is_correct = model_answer == question_data['correct_answer'].upper()
        
        return {
            "id": question_data['id'],
            "correct_answer": question_data['correct_answer'],
            "model_answer": model_answer,
            "reasoning": response.choices[0].message.parsed.reasoning,
            "is_correct": is_correct,
            "response_time": time.time() - start
        }
    except Exception as e:
        logger.error(f"Error on question {question_data.get('id')}: {e}")
        return {
            "id": question_data.get('id', -1),
            "correct_answer": question_data.get('correct_answer', ''),
            "model_answer": "ERROR",
            "reasoning": str(e),
            "is_correct": False,
            "response_time": 0
        }

def main():
    # input_file = "eval/qa_big_beaut_bill.json"
    # input_file = "eval/qa_exe_orders_4.json"
    # input_file = "eval/qa_f1_rules.json"
    # input_file = "eval/qa_tax_rules.json"
    input_file = "eval/dataset/combined_datasets.json"
    model = "gpt-4o"
    max_workers = 5
    
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    # questions = data['questions']
    logger.info(f"Evaluating {len(questions)} questions with {model}")
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_question, q, model): q for q in questions}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            correct_so_far = sum(1 for r in results if r['is_correct'])
            accuracy = (correct_so_far / len(results)) * 100
            
            logger.info(f"Q{result['id']}: {result['model_answer']} vs {result['correct_answer']} "
                       f"({'✓' if result['is_correct'] else '✗'}) | "
                       f"Progress: {len(results)}/{len(questions)} | "
                       f"Accuracy: {accuracy:.1f}%")
    
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / len(results)) * 100
    avg_time = sum(r['response_time'] for r in results) / len(results)
    
    output = {
        "metadata": {
            "dataset": input_file,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(questions),
            "correct": correct_count,
            "accuracy": accuracy,
            "avg_response_time": avg_time
        },
        "results": sorted(results, key=lambda x: x['id'])
    }
    
    os.makedirs("eval/mcq_results", exist_ok=True)
    
    # Extract dataset name from input file path for output filename
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"eval/mcq_results/{dataset_name}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(questions)})")
    print(f"Avg Response Time: {avg_time:.2f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()