"""
Results viewer utility for L-MARS trajectory data.
Provides functionality to view, analyze, and export trajectory results.
"""
import json
import argparse
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lmars.trajectory_tracker import TrajectoryTracker


class ResultsViewer:
    """Utility for viewing and analyzing L-MARS trajectory results."""
    
    def __init__(self, results_dir: str = "results"):
        self.tracker = TrajectoryTracker(results_dir)
    
    def list_runs(self) -> List[Dict[str, str]]:
        """List all available trajectory runs."""
        return self.tracker.list_trajectories()
    
    def view_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """View detailed information about a specific run."""
        trajectory = self.tracker.load_trajectory(run_id)
        if not trajectory:
            return None
        
        return {
            'run_id': trajectory.run_id,
            'start_time': trajectory.start_time,
            'end_time': trajectory.end_time,
            'original_query': trajectory.original_query,
            'duration': self._calculate_duration(trajectory.start_time, trajectory.end_time),
            'steps_count': len(trajectory.steps),
            'human_inputs_count': len(trajectory.human_inputs),
            'final_result': trajectory.final_result,
            'metadata': trajectory.metadata
        }
    
    def view_run_steps(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """View all steps in a trajectory run."""
        trajectory = self.tracker.load_trajectory(run_id)
        if not trajectory:
            return None
        
        steps_info = []
        for step in trajectory.steps:
            steps_info.append({
                'step_id': step.step_id,
                'node_name': step.node_name,
                'timestamp': step.timestamp,
                'duration_seconds': step.duration_seconds,
                'model_calls_count': len(step.model_calls),
                'input_keys': list(step.input_data.keys()) if step.input_data else [],
                'output_keys': list(step.output_data.keys()) if step.output_data else []
            })
        
        return steps_info
    
    def view_model_interactions(self, run_id: str) -> Optional[List[Dict[str, Any]]]:
        """View all model interactions in a run."""
        trajectory = self.tracker.load_trajectory(run_id)
        if not trajectory:
            return None
        
        interactions = []
        for step in trajectory.steps:
            for model_call in step.model_calls:
                interactions.append({
                    'step_name': step.node_name,
                    'timestamp': model_call['timestamp'],
                    'model_name': model_call['model_name'],
                    'input_prompt': model_call['input_prompt'][:200] + '...' if len(model_call['input_prompt']) > 200 else model_call['input_prompt'],
                    'output_response': model_call['output_response'][:200] + '...' if len(model_call['output_response']) > 200 else model_call['output_response'],
                    'token_usage': model_call.get('token_usage', {})
                })
        
        return interactions
    
    def view_human_inputs(self, run_id: str) -> Optional[Dict[str, Any]]:
        """View all human inputs in a run."""
        trajectory = self.tracker.load_trajectory(run_id)
        if not trajectory:
            return None
        
        return trajectory.human_inputs
    
    def export_run(self, run_id: str, output_file: str = None) -> str:
        """Export a run to a detailed JSON file."""
        trajectory = self.tracker.load_trajectory(run_id)
        if not trajectory:
            raise ValueError(f"Run {run_id} not found")
        
        if not output_file:
            output_file = f"detailed_trajectory_{run_id}.json"
        
        # Create detailed export
        export_data = {
            'run_info': {
                'run_id': trajectory.run_id,
                'start_time': trajectory.start_time,
                'end_time': trajectory.end_time,
                'original_query': trajectory.original_query,
                'duration': self._calculate_duration(trajectory.start_time, trajectory.end_time),
                'metadata': trajectory.metadata
            },
            'human_interactions': trajectory.human_inputs,
            'workflow_steps': [],
            'final_result': trajectory.final_result,
            'summary': {
                'total_steps': len(trajectory.steps),
                'total_model_calls': sum(len(step.model_calls) for step in trajectory.steps),
                'total_human_inputs': len(trajectory.human_inputs),
                'average_step_duration': sum(step.duration_seconds for step in trajectory.steps) / len(trajectory.steps) if trajectory.steps else 0
            }
        }
        
        # Add detailed step information
        for step in trajectory.steps:
            step_data = {
                'step_id': step.step_id,
                'node_name': step.node_name,
                'timestamp': step.timestamp,
                'duration_seconds': step.duration_seconds,
                'input_data': step.input_data,
                'output_data': step.output_data,
                'model_calls': step.model_calls
            }
            export_data['workflow_steps'].append(step_data)
        
        # Save to file
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path.absolute())
    
    def _calculate_duration(self, start_time: str, end_time: Optional[str]) -> str:
        """Calculate duration between start and end times."""
        if not end_time:
            return "Unknown (run incomplete)"
        
        try:
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            duration = end - start
            
            total_seconds = int(duration.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            
            if minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        except ValueError:
            return "Unknown"
    
    def print_run_summary(self, run_id: str):
        """Print a formatted summary of a run."""
        run_info = self.view_run(run_id)
        if not run_info:
            print(f"Run {run_id} not found")
            return
        
        print(f"\\n=== L-MARS Trajectory Run: {run_id} ===")
        print(f"Start Time: {run_info['start_time']}")
        print(f"Duration: {run_info['duration']}")
        print(f"Original Query: {run_info['original_query']}")
        print(f"Steps: {run_info['steps_count']}")
        print(f"Human Inputs: {run_info['human_inputs_count']}")
        
        if run_info['final_result']:
            print("\\nFinal Result: Available")
        else:
            print("\\nFinal Result: None (run may be incomplete)")
        
        # Show steps
        steps = self.view_run_steps(run_id)
        if steps:
            print("\\nWorkflow Steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step['node_name']} ({step['duration_seconds']:.2f}s, {step['model_calls_count']} model calls)")


def main():
    """Command-line interface for results viewer."""
    parser = argparse.ArgumentParser(description="View L-MARS trajectory results")
    parser.add_argument("--results-dir", default="results", help="Results directory path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all trajectory runs")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View detailed run information")
    view_parser.add_argument("run_id", help="Run ID to view")
    view_parser.add_argument("--steps", action="store_true", help="Show detailed steps")
    view_parser.add_argument("--models", action="store_true", help="Show model interactions")
    view_parser.add_argument("--human", action="store_true", help="Show human inputs")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export run to detailed JSON")
    export_parser.add_argument("run_id", help="Run ID to export")
    export_parser.add_argument("--output", help="Output file path")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    viewer = ResultsViewer(args.results_dir)
    
    if args.command == "list":
        runs = viewer.list_runs()
        if not runs:
            print("No trajectory runs found")
            return
        
        print(f"\\nFound {len(runs)} trajectory runs:")
        print("-" * 80)
        for run in runs:
            print(f"ID: {run['run_id']}")
            print(f"Time: {run['start_time']}")
            print(f"Query: {run['original_query']}")
            print("-" * 80)
    
    elif args.command == "view":
        if args.steps:
            steps = viewer.view_run_steps(args.run_id)
            if steps:
                print(f"\\nSteps for run {args.run_id}:")
                for step in steps:
                    print(f"- {step['node_name']}: {step['duration_seconds']:.2f}s, {step['model_calls_count']} model calls")
            else:
                print(f"Run {args.run_id} not found")
        
        elif args.models:
            interactions = viewer.view_model_interactions(args.run_id)
            if interactions:
                print(f"\\nModel interactions for run {args.run_id}:")
                for interaction in interactions:
                    print(f"- {interaction['step_name']}: {interaction['model_name']}")
                    print(f"  Input: {interaction['input_prompt']}")
                    print(f"  Output: {interaction['output_response']}")
                    print()
            else:
                print(f"Run {args.run_id} not found")
        
        elif args.human:
            human_inputs = viewer.view_human_inputs(args.run_id)
            if human_inputs:
                print(f"\\nHuman inputs for run {args.run_id}:")
                for key, value in human_inputs.items():
                    print(f"- {key}: {value}")
            else:
                print(f"No human inputs found for run {args.run_id}")
        
        else:
            viewer.print_run_summary(args.run_id)
    
    elif args.command == "export":
        try:
            output_file = viewer.export_run(args.run_id, args.output)
            print(f"Exported run {args.run_id} to {output_file}")
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()