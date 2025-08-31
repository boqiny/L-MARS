"""
Simple MCQ evaluation pipeline for Claude models with JSON mode
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import logging
import anthropic
from dotenv import load_dotenv
import time

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def get_client():
    """Initialize Anthropic client"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    return anthropic.Anthropic(api_key=api_key)

def evaluate_question(question_data, model="claude-3-5-sonnet-20241022"):
    """Evaluate a single MCQ question using Claude with JSON mode"""
    try:
        client = get_client()
        start = time.time()
        
        # Create prompt with JSON format specification
        prompt = f"""You are answering a legal multiple-choice question. 
        
Question: {question_data['mc_question']}

Choices:
A: {question_data['choices']['A']}
B: {question_data['choices']['B']}
C: {question_data['choices']['C']}
D: {question_data['choices']['D']}

Analyze the question and choices carefully, then output your response in the following JSON format:
{{
    "answer": "<single letter A, B, C, or D>",
    "reasoning": "<brief explanation for your choice>"
}}

Important: 
- The "answer" field must contain ONLY a single uppercase letter (A, B, C, or D)
- The "reasoning" field should be a concise explanation (1-3 sentences)
- Output ONLY valid JSON, no additional text"""

        message = client.messages.create(
            model=model,
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0  # Use low temperature for consistency
        )
        
        # Parse the JSON response
        response_text = message.content[0].text if isinstance(message.content, list) else message.content
        
        # Clean the response text to ensure it's valid JSON
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        response_text = response_text.strip()
        
        # Parse JSON
        try:
            parsed_response = json.loads(response_text)
            model_answer = parsed_response.get('answer', '').upper().strip()
            reasoning = parsed_response.get('reasoning', '')
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON for question {question_data.get('id')}: {e}")
            logger.error(f"Response was: {response_text}")
            # Fallback: try to extract answer from text
            model_answer = "ERROR"
            reasoning = f"JSON parsing failed: {response_text[:200]}"
            for letter in ['A', 'B', 'C', 'D']:
                if f'"{letter}"' in response_text or f"'{letter}'" in response_text:
                    model_answer = letter
                    break
        
        # Ensure answer is a single letter
        if len(model_answer) > 1:
            model_answer = model_answer[0]
        
        is_correct = model_answer == question_data['correct_answer'].upper()
        
        return {
            "id": question_data['id'],
            "correct_answer": question_data['correct_answer'],
            "model_answer": model_answer,
            "reasoning": reasoning,
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
    """Main evaluation function"""
    # Configuration
    input_file = "eval/dataset/combined_datasets.json"  # Default dataset
    
    model = "claude-sonnet-4-20250514"  # Latest Claude model
    
    max_workers = 3  # Claude has lower rate limits than OpenAI
    
    # Load questions
    with open(input_file, 'r') as f:
        questions = json.load(f)
    
    logger.info(f"Evaluating {len(questions)} questions with {model}")
    
    # Process questions with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_question, q, model): q for q in questions}
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            # Calculate running accuracy
            correct_so_far = sum(1 for r in results if r['is_correct'])
            accuracy = (correct_so_far / len(results)) * 100
            
            logger.info(f"Q{result['id']}: {result['model_answer']} vs {result['correct_answer']} "
                       f"({'✓' if result['is_correct'] else '✗'}) | "
                       f"Progress: {len(results)}/{len(questions)} | "
                       f"Accuracy: {accuracy:.1f}%")
    
    # Calculate final statistics
    correct_count = sum(1 for r in results if r['is_correct'])
    accuracy = (correct_count / len(results)) * 100 if results else 0
    avg_time = sum(r['response_time'] for r in results) / len(results) if results else 0
    
    # Prepare output
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
    
    # Save results
    os.makedirs("eval/mcq_results", exist_ok=True)
    
    # Extract dataset name from input file path for output filename
    dataset_name = os.path.splitext(os.path.basename(input_file))[0]
    model_short = model.split('-')[-1] if '-' in model else model
    output_file = f"eval/mcq_results/{dataset_name}_claude_{model_short}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Dataset: {input_file}")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(questions)})")
    print(f"Avg Response Time: {avg_time:.2f}s")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}\n")
    
    # Print answer distribution
    answer_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "ERROR": 0}
    for r in results:
        answer = r.get('model_answer', 'ERROR')
        if answer in answer_dist:
            answer_dist[answer] += 1
    
    print("Answer Distribution:")
    for answer, count in answer_dist.items():
        print(f"  {answer}: {count} ({count/len(results)*100:.1f}%)")

if __name__ == "__main__":
    main()