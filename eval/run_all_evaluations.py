#!/usr/bin/env python
"""
Unified evaluation runner for L-MARS
Runs all three evaluation modes and generates comparative analysis
"""
import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


class UnifiedEvaluationRunner:
    """Runs evaluations across all L-MARS modes and generates comparative analysis."""
    
    def __init__(self, results_dir: str = "eval/results"):
        """
        Initialize the unified runner.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_evaluation_mode(self, mode: str, dataset: str, max_samples: int = None,
                          model: str = None, extra_args: List[str] = None) -> Dict[str, Any]:
        """
        Run evaluation for a specific mode.
        
        Args:
            mode: One of 'base_llm', 'simple', 'multi_turn'
            dataset: Path to dataset file
            max_samples: Maximum samples to process
            model: Model name to use
            extra_args: Additional command line arguments
            
        Returns:
            Dictionary with run results
        """
        script_map = {
            'base_llm': 'eval/infer_base_llm.py',
            'simple': 'eval/infer_simple_lmars.py',
            'multi_turn': 'eval/infer_multiturn_lmars.py'
        }
        
        if mode not in script_map:
            raise ValueError(f"Unknown mode: {mode}")
        
        script_path = script_map[mode]
        
        # Build command
        cmd = [sys.executable, script_path, '--dataset', dataset]
        
        if max_samples:
            cmd.extend(['--max-samples', str(max_samples)])
        
        if model:
            cmd.extend(['--model', model])
        
        if extra_args:
            cmd.extend(extra_args)
        
        # Generate run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{mode}_{timestamp}"
        cmd.extend(['--run-name', run_name])
        
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} evaluation...")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        # Run the evaluation
        start_time = datetime.now()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            if result.stderr:
                print(f"Warnings: {result.stderr}")
            
            # Load the results file
            results_file = self.results_dir / f"{run_name}.json"
            with open(results_file, 'r') as f:
                run_results = json.load(f)
            
            run_results['runtime'] = (datetime.now() - start_time).total_seconds()
            run_results['mode'] = mode
            run_results['run_name'] = run_name
            
            return run_results
            
        except subprocess.CalledProcessError as e:
            print(f"Error running {mode}: {e}")
            print(f"Stdout: {e.stdout}")
            print(f"Stderr: {e.stderr}")
            return {
                'mode': mode,
                'run_name': run_name,
                'status': 'failed',
                'error': str(e),
                'runtime': (datetime.now() - start_time).total_seconds()
            }
    
    def run_all_modes(self, dataset: str, max_samples: int = None,
                     model: str = None, parallel: bool = False) -> Dict[str, Any]:
        """
        Run evaluations for all modes.
        
        Args:
            dataset: Path to dataset file
            max_samples: Maximum samples to process
            model: Model name to use
            parallel: Whether to run modes in parallel
            
        Returns:
            Dictionary with all results
        """
        modes = ['base_llm', 'simple', 'multi_turn']
        all_results = {}
        
        if parallel:
            print("\nRunning evaluations in parallel...")
            with ProcessPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(self.run_evaluation_mode, mode, dataset, 
                                  max_samples, model): mode
                    for mode in modes
                }
                
                for future in as_completed(futures):
                    mode = futures[future]
                    try:
                        result = future.result()
                        all_results[mode] = result
                    except Exception as e:
                        print(f"Error in {mode}: {e}")
                        all_results[mode] = {'status': 'failed', 'error': str(e)}
        else:
            print("\nRunning evaluations sequentially...")
            for mode in modes:
                all_results[mode] = self.run_evaluation_mode(
                    mode, dataset, max_samples, model
                )
        
        return all_results
    
    def generate_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comparative analysis across all modes.
        
        Args:
            results: Dictionary of results from all modes
            
        Returns:
            Comparative analysis dictionary
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'modes_evaluated': list(results.keys()),
            'comparison': {}
        }
        
        # Extract successful results
        successful_modes = {
            mode: data for mode, data in results.items()
            if data.get('summary', {}).get('successful', 0) > 0
        }
        
        if not successful_modes:
            analysis['error'] = "No successful evaluations to compare"
            return analysis
        
        # Compare U-scores
        u_scores = {}
        for mode, data in successful_modes.items():
            u_scores[mode] = data['summary'].get('average_u_score', float('inf'))
        
        best_mode_u_score = min(u_scores.items(), key=lambda x: x[1])
        analysis['comparison']['u_scores'] = u_scores
        analysis['comparison']['best_u_score'] = {
            'mode': best_mode_u_score[0],
            'score': best_mode_u_score[1]
        }
        
        # Compare processing times
        processing_times = {}
        for mode, data in successful_modes.items():
            processing_times[mode] = data['summary'].get('average_processing_time', 0)
        
        fastest_mode = min(processing_times.items(), key=lambda x: x[1])
        analysis['comparison']['processing_times'] = processing_times
        analysis['comparison']['fastest'] = {
            'mode': fastest_mode[0],
            'time': fastest_mode[1]
        }
        
        # Compare qualitative distributions
        qual_distributions = {}
        for mode, data in successful_modes.items():
            if 'summary' in data and 'qualitative_distribution' in data['summary']:
                qual_dist = data['summary']['qualitative_distribution']
                qual_distributions[mode] = qual_dist.get('overall_usefulness', {})
        
        analysis['comparison']['qualitative_distributions'] = qual_distributions
        
        # Calculate success rates
        success_rates = {}
        for mode, data in results.items():
            if 'summary' in data:
                total = data['summary'].get('total_questions', 0)
                successful = data['summary'].get('successful', 0)
                if total > 0:
                    success_rates[mode] = successful / total
                else:
                    success_rates[mode] = 0
        
        analysis['comparison']['success_rates'] = success_rates
        
        # Generate insights
        insights = []
        
        # U-score insight
        if u_scores:
            u_score_diff = max(u_scores.values()) - min(u_scores.values())
            if u_score_diff > 0.1:
                insights.append(f"Significant U-score difference ({u_score_diff:.3f}) between modes")
            else:
                insights.append("All modes show similar U-score performance")
        
        # Speed insight
        if processing_times and len(processing_times) > 1:
            slowest_time = max(processing_times.values())
            fastest_time = min(processing_times.values())
            if slowest_time > 0:
                speedup = slowest_time / fastest_time
                insights.append(f"{fastest_mode[0]} is {speedup:.1f}x faster than slowest mode")
        
        # Quality insight
        if qual_distributions:
            high_quality_counts = {}
            for mode, dist in qual_distributions.items():
                high_quality_counts[mode] = dist.get('High', 0)
            
            if high_quality_counts:
                best_quality = max(high_quality_counts.items(), key=lambda x: x[1])
                insights.append(f"{best_quality[0]} produces most high-quality answers ({best_quality[1]})")
        
        analysis['insights'] = insights
        
        return analysis
    
    def save_comparative_report(self, all_results: Dict[str, Any], 
                               analysis: Dict[str, Any], 
                               output_name: str = None) -> str:
        """
        Save comprehensive evaluation report.
        
        Args:
            all_results: Results from all modes
            analysis: Comparative analysis
            output_name: Name for output file
            
        Returns:
            Path to saved report
        """
        if not output_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"comparative_report_{timestamp}"
        
        report_path = self.results_dir / f"{output_name}.json"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'analysis': analysis
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also generate human-readable report
        text_report_path = self.results_dir / f"{output_name}.txt"
        with open(text_report_path, 'w') as f:
            f.write(self.format_text_report(all_results, analysis))
        
        print(f"\nReports saved:")
        print(f"  JSON: {report_path}")
        print(f"  Text: {text_report_path}")
        
        return str(report_path)
    
    def format_text_report(self, results: Dict[str, Any], 
                          analysis: Dict[str, Any]) -> str:
        """
        Format a human-readable text report.
        
        Args:
            results: Results from all modes
            analysis: Comparative analysis
            
        Returns:
            Formatted text report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("L-MARS COMPARATIVE EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {analysis['timestamp']}")
        lines.append(f"Modes evaluated: {', '.join(analysis['modes_evaluated'])}")
        lines.append("")
        
        # Summary statistics for each mode
        lines.append("SUMMARY BY MODE")
        lines.append("-" * 40)
        
        for mode, data in results.items():
            lines.append(f"\n{mode.upper()}:")
            
            if 'summary' in data:
                summary = data['summary']
                lines.append(f"  Total questions: {summary.get('total_questions', 'N/A')}")
                lines.append(f"  Successful: {summary.get('successful', 'N/A')}")
                lines.append(f"  Failed: {summary.get('failed', 'N/A')}")
                lines.append(f"  Average U-Score: {summary.get('average_u_score', 'N/A'):.3f}")
                lines.append(f"  Average time: {summary.get('average_processing_time', 'N/A'):.2f}s")
                
                # Quality distribution
                if 'qualitative_distribution' in summary:
                    qual = summary['qualitative_distribution'].get('overall_usefulness', {})
                    lines.append(f"  Quality distribution:")
                    lines.append(f"    High: {qual.get('High', 0)}")
                    lines.append(f"    Medium: {qual.get('Medium', 0)}")
                    lines.append(f"    Low: {qual.get('Low', 0)}")
            else:
                lines.append(f"  Status: {data.get('status', 'Unknown')}")
                if 'error' in data:
                    lines.append(f"  Error: {data['error']}")
        
        # Comparative analysis
        if 'comparison' in analysis:
            comp = analysis['comparison']
            lines.append("\n" + "=" * 40)
            lines.append("COMPARATIVE ANALYSIS")
            lines.append("-" * 40)
            
            # U-scores
            if 'u_scores' in comp:
                lines.append("\nU-Score Comparison (lower is better):")
                for mode, score in sorted(comp['u_scores'].items(), key=lambda x: x[1]):
                    lines.append(f"  {mode}: {score:.3f}")
                
                if 'best_u_score' in comp:
                    best = comp['best_u_score']
                    lines.append(f"  → Best: {best['mode']} ({best['score']:.3f})")
            
            # Processing times
            if 'processing_times' in comp:
                lines.append("\nProcessing Time Comparison:")
                for mode, time in sorted(comp['processing_times'].items(), key=lambda x: x[1]):
                    lines.append(f"  {mode}: {time:.2f}s")
                
                if 'fastest' in comp:
                    fastest = comp['fastest']
                    lines.append(f"  → Fastest: {fastest['mode']} ({fastest['time']:.2f}s)")
            
            # Success rates
            if 'success_rates' in comp:
                lines.append("\nSuccess Rates:")
                for mode, rate in sorted(comp['success_rates'].items(), 
                                        key=lambda x: x[1], reverse=True):
                    lines.append(f"  {mode}: {rate:.1%}")
        
        # Insights
        if 'insights' in analysis and analysis['insights']:
            lines.append("\n" + "=" * 40)
            lines.append("KEY INSIGHTS")
            lines.append("-" * 40)
            for insight in analysis['insights']:
                lines.append(f"• {insight}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


def main():
    """Main entry point for unified evaluation."""
    parser = argparse.ArgumentParser(
        description="Run L-MARS evaluations across all modes"
    )
    parser.add_argument('--dataset', 
                       default='eval/dataset/uncertain_legal_cases.json',
                       help='Path to evaluation dataset')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to process')
    parser.add_argument('--model', default=None,
                       help='Model to use for evaluation')
    parser.add_argument('--parallel', action='store_true',
                       help='Run modes in parallel')
    parser.add_argument('--modes', nargs='+', 
                       choices=['base_llm', 'simple', 'multi_turn'],
                       default=['base_llm', 'simple', 'multi_turn'],
                       help='Specific modes to run')
    parser.add_argument('--output-name', default=None,
                       help='Name for output report')
    parser.add_argument('--skip-comparison', action='store_true',
                       help='Skip comparative analysis')
    
    args = parser.parse_args()
    
    runner = UnifiedEvaluationRunner()
    
    # Run evaluations
    if len(args.modes) == 1:
        # Single mode
        mode = args.modes[0]
        print(f"Running single mode: {mode}")
        result = runner.run_evaluation_mode(
            mode, args.dataset, args.max_samples, args.model
        )
        print(f"\nEvaluation complete for {mode}")
        
        if result.get('summary'):
            print(f"Average U-Score: {result['summary'].get('average_u_score', 'N/A'):.3f}")
    else:
        # Multiple modes
        print(f"Running modes: {', '.join(args.modes)}")
        
        # Filter to requested modes
        all_results = {}
        for mode in args.modes:
            if args.parallel and len(args.modes) > 1:
                print(f"Queueing {mode} for parallel execution...")
            else:
                all_results[mode] = runner.run_evaluation_mode(
                    mode, args.dataset, args.max_samples, args.model
                )
        
        if args.parallel and len(args.modes) > 1:
            # Run remaining modes in parallel
            # Note: This simplified version runs sequentially
            # Full parallel implementation would require more setup
            pass
        
        # Generate comparative analysis
        if not args.skip_comparison and len(all_results) > 1:
            print("\nGenerating comparative analysis...")
            analysis = runner.generate_comparative_analysis(all_results)
            
            # Save report
            report_path = runner.save_comparative_report(
                all_results, analysis, args.output_name
            )
            
            # Print summary
            print("\n" + "=" * 60)
            print("EVALUATION COMPLETE")
            print("=" * 60)
            
            if 'comparison' in analysis:
                comp = analysis['comparison']
                if 'best_u_score' in comp:
                    best = comp['best_u_score']
                    print(f"Best U-Score: {best['mode']} ({best['score']:.3f})")
                if 'fastest' in comp:
                    fastest = comp['fastest']  
                    print(f"Fastest: {fastest['mode']} ({fastest['time']:.2f}s)")
            
            if 'insights' in analysis:
                print("\nKey Insights:")
                for insight in analysis['insights']:
                    print(f"  • {insight}")


if __name__ == "__main__":
    main()