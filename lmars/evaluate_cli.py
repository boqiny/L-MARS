#!/usr/bin/env python3
"""
CLI for evaluating legal answers independently
"""
import argparse
import sys
from pathlib import Path
import json

from .evaluation import LegalAnswerEvaluator, evaluate_from_log


def evaluate_text(args):
    """Evaluate a text answer."""
    # Read answer text
    if args.file:
        with open(args.file, 'r') as f:
            answer_text = f.read()
    else:
        answer_text = args.text
    
    # Parse sources if provided
    sources = args.sources.split(',') if args.sources else []
    
    # Evaluate
    evaluator = LegalAnswerEvaluator()
    metrics = evaluator.evaluate(answer_text, sources, args.jurisdiction)
    
    # Print report
    print(evaluator.format_report(metrics))
    
    # Save if requested
    if args.save:
        output_file = evaluator.save_evaluation(metrics, args.output)
        print(f"\nEvaluation saved to: {output_file}")
    
    return 0 if metrics.u_score < 0.6 else 1


def evaluate_log(args):
    """Evaluate from a log file."""
    if not Path(args.log).exists():
        print(f"Error: Log file not found: {args.log}")
        return 1
    
    metrics = evaluate_from_log(args.log, save_results=args.save)
    
    if metrics:
        return 0 if metrics.u_score < 0.6 else 1
    return 1


def list_logs(args):
    """List available log files."""
    results_dir = Path("results")
    if not results_dir.exists():
        print("No results directory found")
        return 1
    
    log_files = sorted(results_dir.glob("lmars_log_*.json"))
    
    if not log_files:
        print("No log files found in results directory")
        return 1
    
    print("Available log files:")
    print("-" * 60)
    
    for log_file in log_files[-10:]:  # Show last 10
        # Extract metadata
        with open(log_file, 'r') as f:
            try:
                data = json.load(f)
                query = data.get('query', 'N/A')[:50]
                timestamp = data.get('timestamp', 'N/A')
                mode = data.get('configuration', {}).get('mode', 'N/A')
                
                # Check if evaluation exists
                final_answer = data.get('final_answer', {})
                has_eval = 'evaluation' in final_answer
                
                print(f"\n📁 {log_file.name}")
                print(f"   Query: {query}...")
                print(f"   Mode: {mode}")
                print(f"   Time: {timestamp}")
                print(f"   Evaluated: {'✅' if has_eval else '❌'}")
                
                if has_eval:
                    u_score = final_answer['evaluation']['u_score']
                    print(f"   U-Score: {u_score:.3f}")
                    
            except Exception as e:
                print(f"   Error reading file: {e}")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(log_files)} log files")
    
    return 0


def main():
    """Main entry point for evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate legal answer quality",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Evaluate text command
    text_parser = subparsers.add_parser('text', help='Evaluate a text answer')
    text_group = text_parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument('--text', '-t', help='Answer text to evaluate')
    text_group.add_argument('--file', '-f', help='File containing answer text')
    text_parser.add_argument('--sources', '-s', help='Comma-separated list of source URLs')
    text_parser.add_argument('--jurisdiction', '-j', help='Expected jurisdiction')
    text_parser.add_argument('--save', action='store_true', help='Save evaluation results')
    text_parser.add_argument('--output', '-o', help='Output file name')
    
    # Evaluate log command
    log_parser = subparsers.add_parser('log', help='Evaluate from L-MARS log file')
    log_parser.add_argument('log', help='Path to log file')
    log_parser.add_argument('--save', action='store_true', help='Save evaluation results')
    
    # List logs command
    list_parser = subparsers.add_parser('list', help='List available log files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    if args.command == 'text':
        return evaluate_text(args)
    elif args.command == 'log':
        return evaluate_log(args)
    elif args.command == 'list':
        return list_logs(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())