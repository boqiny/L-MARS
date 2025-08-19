#!/usr/bin/env python3
"""
L-MARS Command Line Interface
Default: Simple mode for quick legal research
Optional: Multi-turn mode for complex queries
"""
import os
import sys
import argparse
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .workflow import create_workflow
from .result_logger import create_logger, close_logger
from .evaluation import evaluate_from_log


def validate_api_keys():
    """Check if required API keys are present."""
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("❌ Error: No API keys found!")
        print("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file")
        return False
    return True


def simple_mode_cli(query: str, model: str = "openai:gpt-4o", verbose: bool = False, 
                    enable_offline_rag: bool = False, enable_courtlistener: bool = False):
    """Execute simple mode workflow from command line."""
    print(f"\n🔍 L-MARS Legal Research (Simple Mode)")
    print(f"📝 Query: {query}")
    
    # Create logger for this session
    logger = create_logger()
    
    # Show enabled sources
    sources = ["Web Search (Serper)"]
    if enable_offline_rag:
        sources.insert(0, "Offline RAG")
    if enable_courtlistener:
        sources.append("CourtListener")
    print(f"🔎 Enabled sources: {', '.join(sources)}\n")
    
    # Create workflow in simple mode
    workflow = create_workflow(
        mode="simple",
        llm_model=model,
        enable_tracking=verbose,
        enable_offline_rag=enable_offline_rag,
        enable_courtlistener=enable_courtlistener
    )
    
    print("🔄 Searching legal sources...")
    
    try:
        result = workflow.run(query)
        
        # Show search results if verbose or if we have them
        if result.get("search_results"):
            print("\n" + "="*60)
            print("🔎 SEARCH RESULTS")
            print("="*60)
            
            for i, search_result in enumerate(result["search_results"], 1):
                print(f"\n📄 Result {i}:")
                print(f"   Source: {search_result.source}")
                print(f"   Title: {search_result.title}")
                print(f"   Content: {search_result.content[:200]}..." if len(search_result.content) > 200 else f"   Content: {search_result.content}")
        
        # Display final answer
        print("\n" + "="*60)
        print("📋 FINAL ANSWER")
        print("="*60 + "\n")
        
        final_answer = result.get("final_answer")
        if final_answer:
            print(f"📌 Answer:\n{final_answer.answer}\n")
            
            if final_answer.key_points:
                print("🔑 Key Points:")
                for point in final_answer.key_points:
                    print(f"  • {point}")
                print()
            
            if final_answer.sources:
                print("📚 Sources:")
                for source in final_answer.sources:
                    print(f"  • {source}")
                print()
            
            if final_answer.disclaimers:
                print("⚠️  Legal Disclaimers:")
                for disclaimer in final_answer.disclaimers:
                    print(f"  • {disclaimer}")
                print()
            
            # Show evaluation if available (would be in result dict)
            if isinstance(result, dict) and 'evaluation_metrics' in result:
                eval_metrics = result['evaluation_metrics']
                print(f"\n📈 Quantitative Evaluation:")
                print(f"  U-Score: {eval_metrics.u_score:.3f}")
                interpretation = "Good" if eval_metrics.u_score < 0.4 else "Moderate" if eval_metrics.u_score < 0.7 else "High uncertainty"
                print(f"  Interpretation: {interpretation}")
                
                # Show qualitative evaluation if available
                if 'combined_evaluation' in result and result['combined_evaluation'].get('qualitative'):
                    qual = result['combined_evaluation']['qualitative']
                    print(f"\n🎯 Qualitative Evaluation (LLM Judge):")
                    print(f"  Factual Accuracy: {qual.factual_accuracy.level}")
                    print(f"  Evidence Grounding: {qual.evidence_grounding.level}")
                    print(f"  Clarity & Reasoning: {qual.clarity_reasoning.level}")
                    print(f"  Uncertainty Awareness: {qual.uncertainty_awareness.level}")
                    print(f"  Overall Usefulness: {qual.overall_usefulness.level}")
        
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if logger:
            logger.log_error("cli_error", str(e))
        return 1
    
    finally:
        # Save and close logger
        log_file = close_logger()
        if log_file:
            print(f"\n📁 Complete log saved to: {log_file}")
    
    return 0


def multi_turn_mode_cli(query: str, model: str = "openai:gpt-4o", judge_model: str = None, 
                        max_iterations: int = 3, enable_offline_rag: bool = False, 
                        enable_courtlistener: bool = False):
    """Execute multi-turn mode workflow with interactive refinement."""
    print(f"\n🔍 L-MARS Legal Research (Multi-Turn Mode)")
    print(f"📝 Query: {query}")
    
    # Create logger for this session
    logger = create_logger()
    
    # Show enabled sources
    sources = ["Web Search (Serper)"]
    if enable_offline_rag:
        sources.insert(0, "Offline RAG")
    if enable_courtlistener:
        sources.append("CourtListener")
    print(f"🔎 Enabled sources: {', '.join(sources)}\n")
    
    # Create workflow in multi-turn mode
    workflow = create_workflow(
        mode="multi_turn",
        llm_model=model,
        judge_model=judge_model or model,
        max_iterations=max_iterations,
        enable_tracking=True,
        enable_offline_rag=enable_offline_rag,
        enable_courtlistener=enable_courtlistener
    )
    
    print("🤔 Analyzing your question...")
    
    try:
        # Initial run
        result = workflow.run(query)
        
        # Check if follow-up questions are needed
        if result.get("needs_user_input") and result.get("follow_up_questions"):
            print("\n📋 I need some clarification to provide better assistance:\n")
            
            user_responses = {}
            for i, question in enumerate(result["follow_up_questions"], 1):
                print(f"{i}. {question.question}")
                print(f"   ({question.reason})")
                response = input(f"   Your answer: ").strip()
                if response:
                    user_responses[f"question_{i}"] = response
                print()
            
            # Continue with user responses
            print("🔄 Searching with refined context...")
            result = workflow.run(query, user_responses)
        
        # Display results
        print("\n" + "="*60)
        print("📋 LEGAL RESEARCH RESULTS")
        print("="*60 + "\n")
        
        final_answer = result.get("final_answer")
        if final_answer:
            print(f"📌 Answer:\n{final_answer.answer}\n")
            
            if final_answer.key_points:
                print("🔑 Key Points:")
                for point in final_answer.key_points:
                    print(f"  • {point}")
                print()
            
            if final_answer.sources:
                print("📚 Sources:")
                for source in final_answer.sources:
                    print(f"  • {source}")
                print()
            
            if final_answer.disclaimers:
                print("⚠️  Legal Disclaimers:")
                for disclaimer in final_answer.disclaimers:
                    print(f"  • {disclaimer}")
                print()
            
            # Show evaluation if available (same as simple mode)
            if isinstance(result, dict) and 'evaluation_metrics' in result:
                eval_metrics = result['evaluation_metrics']
                print(f"\n📈 Quantitative Evaluation:")
                print(f"  U-Score: {eval_metrics.u_score:.3f}")
                interpretation = "Good" if eval_metrics.u_score < 0.4 else "Moderate" if eval_metrics.u_score < 0.7 else "High uncertainty"
                print(f"  Interpretation: {interpretation}")
                
                # Show qualitative evaluation if available
                if 'combined_evaluation' in result and result['combined_evaluation'].get('qualitative'):
                    qual = result['combined_evaluation']['qualitative']
                    print(f"\n🎯 Qualitative Evaluation (LLM Judge):")
                    print(f"  Factual Accuracy: {qual.factual_accuracy.level}")
                    print(f"  Evidence Grounding: {qual.evidence_grounding.level}")
                    print(f"  Clarity & Reasoning: {qual.clarity_reasoning.level}")
                    print(f"  Uncertainty Awareness: {qual.uncertainty_awareness.level}")
                    print(f"  Overall Usefulness: {qual.overall_usefulness.level}")
            
            print(f"\n🔄 Iterations: {result.get('iterations', 1)}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Research interrupted by user")
        if logger:
            logger.log_error("user_interrupt", "Research interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if logger:
            logger.log_error("cli_error", str(e))
        return 1
    
    finally:
        # Save and close logger
        log_file = close_logger()
        if log_file:
            print(f"\n📁 Complete log saved to: {log_file}")
    
    return 0




def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="L-MARS Legal Research System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple mode with online search only (default)
  python -m lmars.cli "Can an F1 student work remotely?"
  
  # Enable offline RAG for local documents
  python -m lmars.cli --offline-rag "Legal question..."
  
  # Enable all sources
  python -m lmars.cli --offline-rag --courtlistener "Complex legal question..."
  
  # Multi-turn mode
  python -m lmars.cli --multi "Complex contract dispute..."
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="Legal question to research"
    )
    
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use multi-turn mode for thorough research"
    )
    
    
    parser.add_argument(
        "--model",
        default="openai:gpt-4o",
        help="LLM model to use (default: openai:gpt-4o)"
    )
    
    parser.add_argument(
        "--judge-model",
        help="Judge model for multi-turn mode (default: same as --model)"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max iterations for multi-turn mode (default: 3)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )
    
    # Search source flags
    parser.add_argument(
        "--offline-rag", "-o",
        action="store_true",
        help="Enable offline RAG search from local documents in 'inputs/' folder"
    )
    
    parser.add_argument(
        "--courtlistener", "-c",
        action="store_true",
        help="Enable CourtListener case law search (requires API token)"
    )
    
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Enable all available search sources"
    )
    
    args = parser.parse_args()
    
    # Handle --all-sources flag
    if args.all_sources:
        args.offline_rag = True
        args.courtlistener = True
    
    # Validate API keys
    if not validate_api_keys():
        return 1
    
    # Handle different modes
    if args.query:
        if args.multi:
            return multi_turn_mode_cli(
                args.query,
                model=args.model,
                judge_model=args.judge_model,
                max_iterations=args.max_iterations,
                enable_offline_rag=args.offline_rag,
                enable_courtlistener=args.courtlistener
            )
        else:
            return simple_mode_cli(
                args.query,
                model=args.model,
                verbose=args.verbose,
                enable_offline_rag=args.offline_rag,
                enable_courtlistener=args.courtlistener
            )
    
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())