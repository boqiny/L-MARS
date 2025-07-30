#!/usr/bin/env python3
"""
Unit tests for the L-MARS trajectory tracking system.
Tests all components of the trajectory tracking functionality.
"""
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lmars.trajectory_tracker import TrajectoryTracker, TrajectoryRun, TrajectoryStep
from lmars.graph import create_legal_mind_graph, LMarsGraph
from test.results_viewer import ResultsViewer


class TestTrajectoryTracker:
    """Test suite for TrajectoryTracker class."""
    
    def __init__(self):
        self.temp_dir = None
        self.tracker = None
    
    def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = TrajectoryTracker(self.temp_dir)
        print(f"✅ Test setup complete - using temp dir: {self.temp_dir}")
    
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("✅ Test cleanup complete")
    
    def test_start_run(self):
        """Test starting a new trajectory run."""
        print("\n🧪 Testing run start functionality...")
        
        query = "Test legal query"
        metadata = {"test": True}
        
        run_id = self.tracker.start_run(query, metadata)
        
        assert run_id is not None, "Run ID should not be None"
        assert self.tracker.current_run is not None, "Current run should be set"
        assert self.tracker.current_run.original_query == query, "Query should match"
        assert self.tracker.current_run.metadata == metadata, "Metadata should match"
        
        print(f"   ✅ Run started with ID: {run_id[:8]}...")
        return run_id
    
    def test_step_tracking(self):
        """Test step tracking functionality."""
        print("\n🧪 Testing step tracking...")
        
        # Start a run first
        run_id = self.test_start_run()
        
        # Test step tracking
        input_data = {"test_input": "value"}
        step_id = self.tracker.start_step("test_node", input_data)
        
        assert step_id is not None, "Step ID should not be None"
        
        # Log a model call
        self.tracker.log_model_call("test-model", "test prompt", "test response", {"tokens": 100})
        
        # End the step
        output_data = {"test_output": "result"}
        self.tracker.end_step(output_data)
        
        assert len(self.tracker.current_run.steps) == 1, "Should have one step"
        step = self.tracker.current_run.steps[0]
        assert step.node_name == "test_node", "Node name should match"
        assert len(step.model_calls) == 1, "Should have one model call"
        
        print("   ✅ Step tracking working correctly")
        return run_id
    
    def test_human_input_logging(self):
        """Test human input logging."""
        print("\n🧪 Testing human input logging...")
        
        run_id = self.test_start_run()
        
        self.tracker.log_human_input("initial_query", "Test question")
        self.tracker.log_human_input("follow_up", {"answer1": "response1"})
        
        assert len(self.tracker.current_run.human_inputs) == 2, "Should have two human inputs"
        
        print("   ✅ Human input logging working correctly")
        return run_id
    
    def test_run_completion_and_saving(self):
        """Test run completion and file saving."""
        print("\n🧪 Testing run completion and saving...")
        
        run_id = self.test_step_tracking()
        
        final_result = {"answer": "Test answer"}
        completed_run_id = self.tracker.end_run(final_result)
        
        assert completed_run_id == run_id, "Run IDs should match"
        assert self.tracker.current_run is None, "Current run should be cleared"
        
        # Check if file was saved
        trajectory_file = Path(self.temp_dir) / f"trajectory_{run_id}.json"
        assert trajectory_file.exists(), "Trajectory file should exist"
        
        # Load and verify saved data
        with open(trajectory_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data['run_id'] == run_id, "Saved run ID should match"
        assert saved_data['original_query'] == "Test legal query", "Saved query should match"
        assert len(saved_data['steps']) == 1, "Should have saved one step"
        
        print(f"   ✅ Run completed and saved to: {trajectory_file.name}")
        return run_id
    
    def test_trajectory_loading(self):
        """Test loading saved trajectories."""
        print("\n🧪 Testing trajectory loading...")
        
        run_id = self.test_run_completion_and_saving()
        
        # Load trajectory
        loaded_trajectory = self.tracker.load_trajectory(run_id)
        
        assert loaded_trajectory is not None, "Should load trajectory"
        assert loaded_trajectory.run_id == run_id, "Run ID should match"
        assert loaded_trajectory.original_query == "Test legal query", "Query should match"
        assert len(loaded_trajectory.steps) == 1, "Should have one step"
        
        print("   ✅ Trajectory loading working correctly")
        return run_id
    
    def test_list_trajectories(self):
        """Test listing trajectories."""
        print("\n🧪 Testing trajectory listing...")
        
        # Create multiple runs
        run_ids = []
        for i in range(3):
            run_id = self.tracker.start_run(f"Test query {i}")
            self.tracker.end_run({"test": i})
            run_ids.append(run_id)
        
        # List trajectories
        trajectories = self.tracker.list_trajectories()
        
        assert len(trajectories) >= 3, "Should have at least 3 trajectories"
        
        # Check structure
        for traj in trajectories[:3]:
            assert 'run_id' in traj, "Should have run_id"
            assert 'start_time' in traj, "Should have start_time"
            assert 'original_query' in traj, "Should have original_query"
        
        print(f"   ✅ Listed {len(trajectories)} trajectories")
        return trajectories
    
    def run_all_tests(self):
        """Run all trajectory tracker tests."""
        print("🚀 Starting TrajectoryTracker tests...")
        
        try:
            self.setup()
            
            self.test_start_run()
            self.test_step_tracking()
            self.test_human_input_logging()
            self.test_run_completion_and_saving()
            self.test_trajectory_loading()
            self.test_list_trajectories()
            
            print("\n🎉 All TrajectoryTracker tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            self.teardown()


class TestResultsViewer:
    """Test suite for ResultsViewer class."""
    
    def __init__(self):
        self.temp_dir = None
        self.tracker = None
        self.viewer = None
    
    def setup(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.tracker = TrajectoryTracker(self.temp_dir)
        self.viewer = ResultsViewer(self.temp_dir)
        
        # Create test data
        self.test_run_id = self.create_test_trajectory()
        print(f"✅ ResultsViewer test setup complete")
    
    def create_test_trajectory(self):
        """Create a test trajectory for testing."""
        run_id = self.tracker.start_run("Test legal question for viewer")
        
        # Add some steps
        step_id1 = self.tracker.start_step("query_processing", {"query": "test"})
        self.tracker.log_model_call("gpt-4o", "Process query", "Query processed")
        self.tracker.end_step({"processed": True})
        
        step_id2 = self.tracker.start_step("search_execution", {"queries": ["test query"]})
        self.tracker.log_model_call("search-tool", "Execute search", "Found results")
        self.tracker.end_step({"results": ["result1", "result2"]})
        
        # Add human input
        self.tracker.log_human_input("initial_query", "Test question")
        
        # Complete run
        self.tracker.end_run({"final_answer": "Test answer"})
        
        return run_id
    
    def teardown(self):
        """Clean up test environment."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("✅ ResultsViewer test cleanup complete")
    
    def test_list_runs(self):
        """Test listing runs via viewer."""
        print("\n🧪 Testing results viewer run listing...")
        
        runs = self.viewer.list_runs()
        
        assert len(runs) >= 1, "Should have at least one run"
        assert any(run['run_id'] == self.test_run_id for run in runs), "Should find test run"
        
        print(f"   ✅ Found {len(runs)} runs via viewer")
    
    def test_view_run(self):
        """Test viewing run details."""
        print("\n🧪 Testing run detail viewing...")
        
        run_info = self.viewer.view_run(self.test_run_id)
        
        assert run_info is not None, "Should return run info"
        assert run_info['run_id'] == self.test_run_id, "Run ID should match"
        assert run_info['steps_count'] == 2, "Should have 2 steps"
        assert run_info['human_inputs_count'] == 1, "Should have 1 human input"
        
        print("   ✅ Run details retrieved correctly")
    
    def test_view_run_steps(self):
        """Test viewing run steps."""
        print("\n🧪 Testing run steps viewing...")
        
        steps = self.viewer.view_run_steps(self.test_run_id)
        
        assert steps is not None, "Should return steps"
        assert len(steps) == 2, "Should have 2 steps"
        assert steps[0]['node_name'] == "query_processing", "First step should be query_processing"
        assert steps[1]['node_name'] == "search_execution", "Second step should be search_execution"
        
        print("   ✅ Run steps retrieved correctly")
    
    def test_view_model_interactions(self):
        """Test viewing model interactions."""
        print("\n🧪 Testing model interactions viewing...")
        
        interactions = self.viewer.view_model_interactions(self.test_run_id)
        
        assert interactions is not None, "Should return interactions"
        assert len(interactions) == 2, "Should have 2 model interactions"
        assert interactions[0]['model_name'] == "gpt-4o", "First interaction should be gpt-4o"
        
        print("   ✅ Model interactions retrieved correctly")
    
    def test_export_run(self):
        """Test run export functionality."""
        print("\n🧪 Testing run export...")
        
        export_file = self.viewer.export_run(self.test_run_id)
        
        assert os.path.exists(export_file), "Export file should exist"
        
        # Load and verify export
        with open(export_file, 'r') as f:
            export_data = json.load(f)
        
        assert 'run_info' in export_data, "Should have run_info"
        assert 'workflow_steps' in export_data, "Should have workflow_steps"
        assert 'summary' in export_data, "Should have summary"
        assert export_data['run_info']['run_id'] == self.test_run_id, "Run ID should match"
        
        print(f"   ✅ Run exported successfully to: {os.path.basename(export_file)}")
    
    def run_all_tests(self):
        """Run all results viewer tests."""
        print("\n🚀 Starting ResultsViewer tests...")
        
        try:
            self.setup()
            
            self.test_list_runs()
            self.test_view_run()
            self.test_view_run_steps()
            self.test_view_model_interactions()
            self.test_export_run()
            
            print("\n🎉 All ResultsViewer tests passed!")
            
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            raise
        finally:
            self.teardown()


class TestLMarsIntegration:
    """Test integration of trajectory tracking with L-MARS workflow."""
    
    def test_mock_workflow(self):
        """Test trajectory tracking integration with mocked LLM."""
        print("\n🚀 Testing L-MARS integration with trajectory tracking...")
        
        # Create temp directory for this test
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Mock the LLM to avoid API calls
            with patch('langchain.chat_models.init_chat_model') as mock_init:
                mock_llm = Mock()
                mock_init.return_value = mock_llm
                
                # Create L-MARS with custom results directory
                lmars = LMarsGraph(enable_tracking=True)
                lmars.tracker.results_dir = Path(temp_dir)
                lmars.tracker.results_dir.mkdir(exist_ok=True)
                
                # Mock agent responses to simulate workflow
                with patch.object(lmars.query_agent, 'generate_followup_questions') as mock_followup, \
                     patch.object(lmars.query_agent, 'generate_search_queries') as mock_queries, \
                     patch.object(lmars.search_agent, 'execute_search') as mock_search, \
                     patch.object(lmars.judge_agent, 'evaluate_results') as mock_judge, \
                     patch.object(lmars.summary_agent, 'generate_final_answer') as mock_summary:
                    
                    # Set up mock responses
                    mock_followup.return_value = []  # No follow-up questions
                    mock_queries.return_value = [Mock(query="test query")]
                    mock_search.return_value = [Mock(title="test result")]
                    mock_judge.return_value = Mock(is_sufficient=True)
                    mock_summary.return_value = Mock(
                        answer="Test answer",
                        key_points=["Point 1"],
                        sources=["Source 1"],
                        disclaimers=["Disclaimer 1"]
                    )
                    
                    # Run a test query
                    test_query = "Test legal question for integration"
                    result = lmars.invoke(test_query)
                    
                    # Verify trajectory was created
                    trajectory_files = list(Path(temp_dir).glob("trajectory_*.json"))
                    assert len(trajectory_files) == 1, "Should create one trajectory file"
                    
                    # Load and verify trajectory
                    with open(trajectory_files[0], 'r') as f:
                        trajectory_data = json.load(f)
                    
                    assert trajectory_data['original_query'] == test_query, "Query should match"
                    assert len(trajectory_data['steps']) > 0, "Should have workflow steps"
                    assert 'human_inputs' in trajectory_data, "Should have human inputs"
                    
                    print(f"   ✅ Integration test passed - trajectory saved with {len(trajectory_data['steps'])} steps")
        
        finally:
            # Clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    
    def run_all_tests(self):
        """Run all integration tests."""
        try:
            self.test_mock_workflow()
            print("\n🎉 All integration tests passed!")
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            raise


def main():
    """Run all tests."""
    print("=" * 80)
    print("🧪 L-MARS Trajectory System Test Suite")
    print("=" * 80)
    
    total_tests = 0
    passed_tests = 0
    
    # Test TrajectoryTracker
    try:
        tracker_tests = TestTrajectoryTracker()
        tracker_tests.run_all_tests()
        passed_tests += 1
    except Exception as e:
        print(f"❌ TrajectoryTracker tests failed: {e}")
    total_tests += 1
    
    # Test ResultsViewer
    try:
        viewer_tests = TestResultsViewer()
        viewer_tests.run_all_tests()
        passed_tests += 1
    except Exception as e:
        print(f"❌ ResultsViewer tests failed: {e}")
    total_tests += 1
    
    # Test Integration
    try:
        integration_tests = TestLMarsIntegration()
        integration_tests.run_all_tests()
        passed_tests += 1
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
    total_tests += 1
    
    # Final results
    print("\n" + "=" * 80)
    print(f"🏁 Test Results: {passed_tests}/{total_tests} test suites passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Trajectory system is working correctly.")
        return 0
    else:
        print(f"❌ {total_tests - passed_tests} test suite(s) failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)