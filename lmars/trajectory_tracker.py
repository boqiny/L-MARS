"""
Trajectory tracking system for L-MARS workflow.
Captures complete run trajectories including model inputs/outputs and human interactions.
"""
import json
import uuid
import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from langchain_core.messages import BaseMessage


@dataclass
class TrajectoryStep:
    """Single step in the workflow trajectory."""
    step_id: str
    node_name: str
    timestamp: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    model_calls: List[Dict[str, Any]]
    duration_seconds: float


@dataclass
class TrajectoryRun:
    """Complete trajectory for a single run."""
    run_id: str
    start_time: str
    end_time: Optional[str]
    original_query: str
    human_inputs: Dict[str, Any]
    steps: List[TrajectoryStep]
    final_result: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class TrajectoryTracker:
    """Tracks and saves complete workflow trajectories."""
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.current_run: Optional[TrajectoryRun] = None
        self.step_start_time: Optional[datetime.datetime] = None
    
    def start_run(self, original_query: str, metadata: Dict[str, Any] = None) -> str:
        """Start tracking a new run."""
        run_id = str(uuid.uuid4())
        start_time = datetime.datetime.now().isoformat()
        
        self.current_run = TrajectoryRun(
            run_id=run_id,
            start_time=start_time,
            end_time=None,
            original_query=original_query,
            human_inputs={},
            steps=[],
            final_result=None,
            metadata=metadata or {}
        )
        
        return run_id
    
    def start_step(self, node_name: str, input_data: Dict[str, Any]) -> str:
        """Start tracking a workflow step."""
        if not self.current_run:
            raise ValueError("No active run. Call start_run() first.")
        
        step_id = str(uuid.uuid4())
        self.step_start_time = datetime.datetime.now()
        
        # Store current step info for completion
        self._current_step_info = {
            'step_id': step_id,
            'node_name': node_name,
            'input_data': self._serialize_data(input_data),
            'model_calls': []
        }
        
        return step_id
    
    def log_model_call(self, model_name: str, input_prompt: str, output_response: str, 
                      token_usage: Dict[str, int] = None):
        """Log a model API call."""
        if not hasattr(self, '_current_step_info'):
            return
        
        model_call = {
            'timestamp': datetime.datetime.now().isoformat(),
            'model_name': model_name,
            'input_prompt': input_prompt,
            'output_response': output_response,
            'token_usage': token_usage or {}
        }
        
        self._current_step_info['model_calls'].append(model_call)
    
    def end_step(self, output_data: Dict[str, Any]):
        """Complete tracking of a workflow step."""
        if not self.current_run or not hasattr(self, '_current_step_info'):
            return
        
        end_time = datetime.datetime.now()
        duration = (end_time - self.step_start_time).total_seconds()
        
        step = TrajectoryStep(
            step_id=self._current_step_info['step_id'],
            node_name=self._current_step_info['node_name'],
            timestamp=self.step_start_time.isoformat(),
            input_data=self._current_step_info['input_data'],
            output_data=self._serialize_data(output_data),
            model_calls=self._current_step_info['model_calls'],
            duration_seconds=duration
        )
        
        self.current_run.steps.append(step)
        delattr(self, '_current_step_info')
    
    def log_human_input(self, input_type: str, input_data: Any):
        """Log human input during the workflow."""
        if not self.current_run:
            return
        
        timestamp = datetime.datetime.now().isoformat()
        self.current_run.human_inputs[f"{input_type}_{timestamp}"] = {
            'type': input_type,
            'data': self._serialize_data(input_data),
            'timestamp': timestamp
        }
    
    def end_run(self, final_result: Dict[str, Any] = None):
        """Complete and save the trajectory run."""
        if not self.current_run:
            return
        
        self.current_run.end_time = datetime.datetime.now().isoformat()
        self.current_run.final_result = self._serialize_data(final_result) if final_result else None
        
        # Save to file
        self._save_trajectory()
        
        # Reset current run
        run_id = self.current_run.run_id
        self.current_run = None
        
        return run_id
    
    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if data is None:
            return None
        
        if isinstance(data, BaseMessage):
            return {
                'type': data.__class__.__name__,
                'content': data.content,
                'additional_kwargs': getattr(data, 'additional_kwargs', {})
            }
        
        if isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        
        if isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        
        if hasattr(data, '__dict__'):
            return {
                'class': data.__class__.__name__,
                'data': {key: self._serialize_data(value) for key, value in data.__dict__.items()}
            }
        
        # For basic types (str, int, float, bool)
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)
    
    def _save_trajectory(self):
        """Save trajectory to JSON file."""
        if not self.current_run:
            return
        
        filename = f"trajectory_{self.current_run.run_id}.json"
        filepath = self.results_dir / filename
        
        trajectory_dict = asdict(self.current_run)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trajectory_dict, f, indent=2, ensure_ascii=False)
    
    def load_trajectory(self, run_id: str) -> Optional[TrajectoryRun]:
        """Load a saved trajectory by run ID."""
        filename = f"trajectory_{run_id}.json"
        filepath = self.results_dir / filename
        
        if not filepath.exists():
            return None
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert back to TrajectoryRun object
        steps = [TrajectoryStep(**step) for step in data['steps']]
        data['steps'] = steps
        
        return TrajectoryRun(**data)
    
    def list_trajectories(self) -> List[Dict[str, str]]:
        """List all saved trajectories."""
        trajectories = []
        
        for filepath in self.results_dir.glob("trajectory_*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                trajectories.append({
                    'run_id': data['run_id'],
                    'start_time': data['start_time'],
                    'original_query': data['original_query'][:100] + '...' if len(data['original_query']) > 100 else data['original_query']
                })
            except (json.JSONDecodeError, KeyError):
                continue
        
        return sorted(trajectories, key=lambda x: x['start_time'], reverse=True)