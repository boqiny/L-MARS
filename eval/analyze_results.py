#!/usr/bin/env python
"""
Analysis utilities for L-MARS evaluation results
Provides tools for analyzing, comparing, and visualizing evaluation metrics
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import statistics
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ResultsAnalyzer:
    """Analyzer for L-MARS evaluation results."""
    
    def __init__(self, results_dir: str = "eval/results"):
        """
        Initialize the analyzer.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        
    def load_results(self, pattern: str = "*.json") -> Dict[str, Any]:
        """
        Load all results matching pattern.
        
        Args:
            pattern: Glob pattern for result files
            
        Returns:
            Dictionary of loaded results
        """
        result_files = list(self.results_dir.glob(pattern))
        
        for file_path in result_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.results[file_path.stem] = data
        
        print(f"Loaded {len(self.results)} result files")
        return self.results
    
    def analyze_single_run(self, run_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single evaluation run.
        
        Args:
            run_data: Data from a single evaluation run
            
        Returns:
            Analysis dictionary
        """
        analysis = {
            'summary': run_data.get('summary', {}),
            'metrics': {}
        }
        
        if 'results' not in run_data:
            return analysis
        
        results = run_data['results']
        successful = [r for r in results if r.get('status') == 'success']
        
        if not successful:
            return analysis
        
        # Analyze U-scores
        u_scores = [r['evaluation']['quantitative']['u_score'] for r in successful
                   if 'evaluation' in r and 'quantitative' in r['evaluation']]
        
        if u_scores:
            analysis['metrics']['u_score'] = {
                'mean': statistics.mean(u_scores),
                'median': statistics.median(u_scores),
                'stdev': statistics.stdev(u_scores) if len(u_scores) > 1 else 0,
                'min': min(u_scores),
                'max': max(u_scores),
                'quartiles': {
                    'q1': statistics.quantiles(u_scores, n=4)[0] if len(u_scores) > 3 else u_scores[0],
                    'q2': statistics.median(u_scores),
                    'q3': statistics.quantiles(u_scores, n=4)[2] if len(u_scores) > 3 else u_scores[-1]
                }
            }
        
        # Analyze component scores
        components = ['hedging_score', 'temporal_vagueness', 'citation_score', 
                     'jurisdiction_score', 'decisiveness_score']
        
        for component in components:
            scores = [r['evaluation']['quantitative'][component] for r in successful
                     if 'evaluation' in r and 'quantitative' in r['evaluation'] 
                     and component in r['evaluation']['quantitative']]
            
            if scores:
                analysis['metrics'][component] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'stdev': statistics.stdev(scores) if len(scores) > 1 else 0
                }
        
        # Analyze qualitative metrics
        qual_metrics = defaultdict(lambda: defaultdict(int))
        for r in successful:
            if 'evaluation' in r and 'qualitative' in r['evaluation']:
                qual = r['evaluation']['qualitative']
                for metric in ['factual_accuracy', 'evidence_grounding', 
                             'clarity_reasoning', 'uncertainty_awareness', 
                             'overall_usefulness']:
                    if metric in qual:
                        level = qual[metric]
                        qual_metrics[metric][level] += 1
        
        analysis['metrics']['qualitative'] = dict(qual_metrics)
        
        # Analyze processing times
        times = [r.get('processing_time', 0) for r in successful]
        if times:
            analysis['metrics']['processing_time'] = {
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'total': sum(times),
                'min': min(times),
                'max': max(times)
            }
        
        # Question-level performance
        analysis['question_performance'] = []
        for r in results[:10]:  # Limit to first 10 for brevity
            if r.get('status') == 'success':
                perf = {
                    'id': r.get('id'),
                    'u_score': r['evaluation']['quantitative']['u_score'],
                    'overall_usefulness': r['evaluation']['qualitative'].get('overall_usefulness'),
                    'time': r.get('processing_time', 0)
                }
                analysis['question_performance'].append(perf)
        
        return analysis
    
    def compare_runs(self, run_names: List[str] = None) -> Dict[str, Any]:
        """
        Compare multiple evaluation runs.
        
        Args:
            run_names: List of run names to compare (None for all)
            
        Returns:
            Comparison dictionary
        """
        if not run_names:
            run_names = list(self.results.keys())
        
        comparison = {
            'runs': run_names,
            'metrics': {}
        }
        
        # Collect metrics from each run
        run_analyses = {}
        for name in run_names:
            if name in self.results:
                run_analyses[name] = self.analyze_single_run(self.results[name])
        
        # Compare U-scores
        u_score_comparison = {}
        for name, analysis in run_analyses.items():
            if 'u_score' in analysis.get('metrics', {}):
                u_score_comparison[name] = analysis['metrics']['u_score']['mean']
        
        if u_score_comparison:
            best_run = min(u_score_comparison.items(), key=lambda x: x[1])
            worst_run = max(u_score_comparison.items(), key=lambda x: x[1])
            
            comparison['metrics']['u_score'] = {
                'by_run': u_score_comparison,
                'best': {'run': best_run[0], 'score': best_run[1]},
                'worst': {'run': worst_run[0], 'score': worst_run[1]},
                'spread': worst_run[1] - best_run[1]
            }
        
        # Compare processing times
        time_comparison = {}
        for name, analysis in run_analyses.items():
            if 'processing_time' in analysis.get('metrics', {}):
                time_comparison[name] = analysis['metrics']['processing_time']['mean']
        
        if time_comparison:
            fastest = min(time_comparison.items(), key=lambda x: x[1])
            slowest = max(time_comparison.items(), key=lambda x: x[1])
            
            comparison['metrics']['processing_time'] = {
                'by_run': time_comparison,
                'fastest': {'run': fastest[0], 'time': fastest[1]},
                'slowest': {'run': slowest[0], 'time': slowest[1]},
                'speedup': slowest[1] / fastest[1] if fastest[1] > 0 else 1
            }
        
        # Compare success rates
        success_rates = {}
        for name in run_names:
            if name in self.results and 'summary' in self.results[name]:
                summary = self.results[name]['summary']
                total = summary.get('total_questions', 0)
                successful = summary.get('successful', 0)
                if total > 0:
                    success_rates[name] = successful / total
        
        if success_rates:
            comparison['metrics']['success_rate'] = {
                'by_run': success_rates,
                'best': max(success_rates.items(), key=lambda x: x[1]),
                'worst': min(success_rates.items(), key=lambda x: x[1])
            }
        
        # Compare qualitative distributions
        qual_comparison = defaultdict(dict)
        for name, analysis in run_analyses.items():
            if 'qualitative' in analysis.get('metrics', {}):
                qual = analysis['metrics']['qualitative']
                for metric in qual:
                    high_count = qual[metric].get('High', 0)
                    total_count = sum(qual[metric].values())
                    if total_count > 0:
                        qual_comparison[metric][name] = high_count / total_count
        
        if qual_comparison:
            comparison['metrics']['qualitative_high_rates'] = dict(qual_comparison)
        
        return comparison
    
    def find_outliers(self, run_name: str, threshold: float = 2.0) -> List[Dict[str, Any]]:
        """
        Find outlier questions in a run based on U-score.
        
        Args:
            run_name: Name of the run to analyze
            threshold: Number of standard deviations for outlier detection
            
        Returns:
            List of outlier questions
        """
        if run_name not in self.results:
            return []
        
        run_data = self.results[run_name]
        if 'results' not in run_data:
            return []
        
        results = run_data['results']
        successful = [r for r in results if r.get('status') == 'success']
        
        if len(successful) < 3:
            return []
        
        # Calculate U-score statistics
        u_scores = [r['evaluation']['quantitative']['u_score'] for r in successful]
        mean = statistics.mean(u_scores)
        stdev = statistics.stdev(u_scores)
        
        outliers = []
        for r in successful:
            u_score = r['evaluation']['quantitative']['u_score']
            z_score = (u_score - mean) / stdev if stdev > 0 else 0
            
            if abs(z_score) > threshold:
                outliers.append({
                    'id': r.get('id'),
                    'question': r.get('question', '')[:100],
                    'u_score': u_score,
                    'z_score': z_score,
                    'type': 'high_uncertainty' if z_score > 0 else 'low_uncertainty'
                })
        
        return sorted(outliers, key=lambda x: abs(x['z_score']), reverse=True)
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive analysis report.
        
        Args:
            output_file: Path to save report (None for stdout)
            
        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("L-MARS EVALUATION ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Total runs analyzed: {len(self.results)}")
        lines.append("")
        
        # Individual run analyses
        lines.append("INDIVIDUAL RUN ANALYSES")
        lines.append("-" * 40)
        
        for run_name, run_data in self.results.items():
            lines.append(f"\n{run_name}:")
            
            if 'summary' in run_data:
                summary = run_data['summary']
                lines.append(f"  Model: {summary.get('model', 'Unknown')}")
                lines.append(f"  Mode: {summary.get('mode', 'Unknown')}")
                lines.append(f"  Questions: {summary.get('total_questions', 0)}")
                lines.append(f"  Success rate: {summary.get('successful', 0)}/{summary.get('total_questions', 0)}")
                lines.append(f"  Avg U-Score: {summary.get('average_u_score', 'N/A'):.3f}")
                lines.append(f"  Avg Time: {summary.get('average_processing_time', 'N/A'):.2f}s")
            
            # Detailed analysis
            analysis = self.analyze_single_run(run_data)
            if 'u_score' in analysis.get('metrics', {}):
                u_metrics = analysis['metrics']['u_score']
                lines.append(f"  U-Score distribution:")
                lines.append(f"    Mean: {u_metrics['mean']:.3f} ± {u_metrics['stdev']:.3f}")
                lines.append(f"    Median: {u_metrics['median']:.3f}")
                lines.append(f"    Range: [{u_metrics['min']:.3f}, {u_metrics['max']:.3f}]")
        
        # Comparative analysis
        if len(self.results) > 1:
            lines.append("\n" + "=" * 40)
            lines.append("COMPARATIVE ANALYSIS")
            lines.append("-" * 40)
            
            comparison = self.compare_runs()
            
            # U-score comparison
            if 'u_score' in comparison.get('metrics', {}):
                u_comp = comparison['metrics']['u_score']
                lines.append("\nU-Score Comparison:")
                for run, score in sorted(u_comp['by_run'].items(), key=lambda x: x[1]):
                    lines.append(f"  {run}: {score:.3f}")
                lines.append(f"  Best: {u_comp['best']['run']} ({u_comp['best']['score']:.3f})")
                lines.append(f"  Spread: {u_comp['spread']:.3f}")
            
            # Time comparison
            if 'processing_time' in comparison.get('metrics', {}):
                time_comp = comparison['metrics']['processing_time']
                lines.append("\nProcessing Time Comparison:")
                for run, time in sorted(time_comp['by_run'].items(), key=lambda x: x[1]):
                    lines.append(f"  {run}: {time:.2f}s")
                lines.append(f"  Speedup: {time_comp['speedup']:.1f}x")
            
            # Success rate comparison
            if 'success_rate' in comparison.get('metrics', {}):
                rate_comp = comparison['metrics']['success_rate']
                lines.append("\nSuccess Rate Comparison:")
                for run, rate in sorted(rate_comp['by_run'].items(), 
                                       key=lambda x: x[1], reverse=True):
                    lines.append(f"  {run}: {rate:.1%}")
        
        # Find outliers in each run
        lines.append("\n" + "=" * 40)
        lines.append("OUTLIER ANALYSIS")
        lines.append("-" * 40)
        
        for run_name in list(self.results.keys())[:3]:  # Limit to first 3 runs
            outliers = self.find_outliers(run_name)
            if outliers:
                lines.append(f"\n{run_name} outliers:")
                for outlier in outliers[:3]:  # Show top 3 outliers
                    lines.append(f"  Q{outlier['id']}: U={outlier['u_score']:.3f} "
                               f"(z={outlier['z_score']:.2f}) - {outlier['type']}")
        
        lines.append("\n" + "=" * 80)
        
        report_text = "\n".join(lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        else:
            print(report_text)
        
        return report_text
    
    def export_metrics_csv(self, output_file: str = "metrics.csv"):
        """
        Export metrics to CSV for further analysis.
        
        Args:
            output_file: Path to output CSV file
        """
        import csv
        
        rows = []
        headers = ['run_name', 'mode', 'model', 'question_id', 'u_score', 
                  'hedging', 'temporal', 'citation', 'jurisdiction', 
                  'decisiveness', 'overall_usefulness', 'processing_time']
        
        for run_name, run_data in self.results.items():
            if 'results' not in run_data:
                continue
            
            summary = run_data.get('summary', {})
            mode = summary.get('mode', 'unknown')
            model = summary.get('model', 'unknown')
            
            for r in run_data['results']:
                if r.get('status') != 'success':
                    continue
                
                row = {
                    'run_name': run_name,
                    'mode': mode,
                    'model': model,
                    'question_id': r.get('id'),
                    'u_score': r['evaluation']['quantitative']['u_score'],
                    'hedging': r['evaluation']['quantitative']['hedging_score'],
                    'temporal': r['evaluation']['quantitative']['temporal_vagueness'],
                    'citation': r['evaluation']['quantitative']['citation_score'],
                    'jurisdiction': r['evaluation']['quantitative']['jurisdiction_score'],
                    'decisiveness': r['evaluation']['quantitative']['decisiveness_score'],
                    'overall_usefulness': r['evaluation']['qualitative'].get('overall_usefulness'),
                    'processing_time': r.get('processing_time', 0)
                }
                rows.append(row)
        
        output_path = self.results_dir / output_file
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Metrics exported to: {output_path}")
        print(f"Total rows: {len(rows)}")


def main():
    """Main entry point for results analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze L-MARS evaluation results"
    )
    parser.add_argument('--results-dir', default='eval/results',
                       help='Directory containing result files')
    parser.add_argument('--pattern', default='*.json',
                       help='Pattern for result files to analyze')
    parser.add_argument('--compare', nargs='+',
                       help='Specific runs to compare')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export metrics to CSV')
    parser.add_argument('--find-outliers', metavar='RUN_NAME',
                       help='Find outliers in specific run')
    parser.add_argument('--output', default=None,
                       help='Output file for report')
    
    args = parser.parse_args()
    
    analyzer = ResultsAnalyzer(args.results_dir)
    analyzer.load_results(args.pattern)
    
    if not analyzer.results:
        print("No results found to analyze")
        return
    
    if args.find_outliers:
        outliers = analyzer.find_outliers(args.find_outliers)
        if outliers:
            print(f"\nOutliers in {args.find_outliers}:")
            for outlier in outliers:
                print(f"  Q{outlier['id']}: U={outlier['u_score']:.3f} "
                     f"(z={outlier['z_score']:.2f}) - {outlier['type']}")
                print(f"    {outlier['question']}")
        else:
            print(f"No outliers found in {args.find_outliers}")
    
    elif args.compare:
        comparison = analyzer.compare_runs(args.compare)
        print(json.dumps(comparison, indent=2))
    
    else:
        analyzer.generate_report(args.output)
    
    if args.export_csv:
        analyzer.export_metrics_csv()


if __name__ == "__main__":
    main()