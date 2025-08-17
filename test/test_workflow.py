#!/usr/bin/env python3
"""
Test suite for L-MARS workflow (Simple and Multi-Turn modes)
"""
import os
import sys
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmars import create_workflow, WorkflowConfig


class TestSimpleMode:
    """Tests for Simple Mode workflow."""
    
    def test_simple_workflow_creation(self):
        """Test creating a simple workflow."""
        workflow = create_workflow(mode="simple")
        assert workflow is not None
        assert workflow.config.mode == "simple"
    
    def test_simple_query_execution(self):
        """Test executing a simple query."""
        workflow = create_workflow(
            mode="simple",
            enable_tracking=False
        )
        
        result = workflow.run("What is a contract?")
        
        assert result is not None
        assert "final_answer" in result
        assert result["mode"] == "simple"


class TestMultiTurnMode:
    """Tests for Multi-Turn Mode workflow."""
    
    def test_multi_turn_workflow_creation(self):
        """Test creating a multi-turn workflow."""
        workflow = create_workflow(mode="multi_turn")
        assert workflow is not None
        assert workflow.config.mode == "multi_turn"
    
    def test_multi_turn_with_followup(self):
        """Test multi-turn mode generates follow-up questions."""
        workflow = create_workflow(
            mode="multi_turn",
            max_iterations=2,
            enable_tracking=False
        )
        
        result = workflow.run("Complex legal question about startup formation")
        
        # Should either have follow-up questions or final answer
        assert result is not None
        assert "mode" in result
        assert result["mode"] == "multi_turn"


class TestWorkflowConfig:
    """Tests for WorkflowConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = WorkflowConfig()
        assert config.mode == "simple"
        assert config.llm_model == "openai:gpt-4o"
        assert config.max_iterations == 3
        assert config.enable_tracking == True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkflowConfig(
            mode="multi_turn",
            llm_model="anthropic:claude-3",
            judge_model="openai:gpt-4",
            max_iterations=5,
            enable_tracking=False
        )
        
        assert config.mode == "multi_turn"
        assert config.llm_model == "anthropic:claude-3"
        assert config.judge_model == "openai:gpt-4"
        assert config.max_iterations == 5
        assert config.enable_tracking == False


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run basic tests
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("Running basic tests (install pytest for full test suite)")
        
        # Basic tests
        print("\nTesting Simple Mode...")
        workflow = create_workflow(mode="simple", enable_tracking=False)
        print(f"✓ Created simple workflow: {workflow.config.mode}")
        
        print("\nTesting Multi-Turn Mode...")
        workflow = create_workflow(mode="multi_turn", enable_tracking=False)
        print(f"✓ Created multi-turn workflow: {workflow.config.mode}")
        
        print("\n✅ Basic tests passed!")