"""
LLM-as-Judge Evaluator for Legal Answers
Provides qualitative assessment alongside quantitative U-score
"""
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
import json
from datetime import datetime
from pathlib import Path


class MetricAssessment(BaseModel):
    """Assessment for a single evaluation metric."""
    level: Literal["Low", "Medium", "High"] = Field(
        description="Quality level for this metric"
    )
    justification: str = Field(
        description="1-2 sentence justification for the assessment"
    )


class LegalAnswerJudgment(BaseModel):
    """Complete LLM judgment of a legal answer."""
    factual_accuracy: MetricAssessment = Field(
        description="Assessment of factual accuracy and alignment with actual law"
    )
    evidence_grounding: MetricAssessment = Field(
        description="Assessment of source citations and evidence support"
    )
    clarity_reasoning: MetricAssessment = Field(
        description="Assessment of reasoning clarity and logical coherence"
    )
    uncertainty_awareness: MetricAssessment = Field(
        description="Assessment of how well uncertainty and limitations are acknowledged"
    )
    overall_usefulness: MetricAssessment = Field(
        description="Assessment of overall usefulness for legal information seekers"
    )
    summary: str = Field(
        description="Brief overall assessment summary (2-3 sentences)",
        default=""
    )
    strengths: List[str] = Field(
        description="Key strengths of the answer",
        default=[]
    )
    weaknesses: List[str] = Field(
        description="Key weaknesses or areas for improvement",
        default=[]
    )


class LLMJudgeEvaluator:
    """LLM-based judge for evaluating legal answer quality."""
    
    def __init__(self, model: str = "openai:gpt-4o"):
        """
        Initialize the LLM judge evaluator.
        
        Args:
            model: LLM model to use for evaluation
        """
        self.llm = init_chat_model(model)
    
    def evaluate(self, 
                 question: str, 
                 answer: str, 
                 sources: List[str] = None,
                 search_results: List[Dict[str, Any]] = None) -> LegalAnswerJudgment:
        """
        Evaluate a legal answer using LLM-as-judge.
        
        Args:
            question: The original legal question
            answer: The generated answer to evaluate
            sources: List of sources cited in the answer
            search_results: Optional search results used to generate the answer
            
        Returns:
            LegalAnswerJudgment with qualitative assessments
        """
        # Prepare sources context
        sources_context = ""
        if sources:
            sources_context = "\n\nCited Sources:\n" + "\n".join([f"- {s}" for s in sources])
        
        # Prepare search results context if available
        search_context = ""
        if search_results:
            search_context = "\n\nSearch Results Used (for reference):\n"
            for i, result in enumerate(search_results[:5], 1):  # Limit to 5 for context
                search_context += f"{i}. Source: {result.get('source', 'Unknown')}\n"
                search_context += f"   Content: {result.get('content', '')[:200]}...\n\n"
        
        prompt = f"""You are a legal domain evaluator tasked with reviewing a model's answer to a legal research question.

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}
{sources_context}
{search_context}

Please assess the response along the following metrics. For each metric, provide:
1. A level assessment: Low, Medium, or High
2. A 1-2 sentence justification

EVALUATION METRICS:

1. **Factual Accuracy**
   - Does the answer align with actual law, case rulings, or authoritative regulations?
   - High = No factual errors, directly supported by authoritative sources
   - Medium = Mostly correct but contains minor inaccuracies, omissions, or outdated phrasing
   - Low = Major errors, misinterpretation, or unsupported claims

2. **Evidence Grounding**
   - Is the answer supported by cited legal sources (e.g., statutes, case law, official agencies)?
   - High = Citations are authoritative (.gov, .court, .edu) and clearly tied to claims
   - Medium = Citations are partially relevant, incomplete, or weakly connected
   - Low = No citations, or only non-authoritative sources (forums, blogs)

3. **Clarity & Reasoning**
   - Is the reasoning chain clear, logically coherent, and legally sound?
   - High = Clear explanation, step-by-step logic, connects law to answer
   - Medium = Reasoning somewhat unclear or oversimplified, but still understandable
   - Low = Disorganized, confusing, or legally incoherent reasoning

4. **Uncertainty Awareness**
   - Does the answer acknowledge unresolved legal debates, jurisdictional conflicts, or evolving laws?
   - High = Explicitly flags uncertainty, jurisdictional variations, and limits of knowledge
   - Medium = Implicitly hedges but not very clear about limitations
   - Low = Overconfident, ignores uncertainty, or states potentially variable laws as universal facts

5. **Overall Usefulness**
   - Would this answer be useful for a lawyer, student, or layperson seeking legal information?
   - High = Reliable, actionable, well-structured, appropriate disclaimers
   - Medium = Partially helpful but needs verification or refinement
   - Low = Misleading, confusing, or unhelpful

Additionally, provide:
- A brief summary of the overall assessment (2-3 sentences)
- 2-3 key strengths of the answer
- 2-3 key weaknesses or areas for improvement

Focus on substantive legal quality rather than stylistic preferences."""
        
        try:
            # Use structured output
            structured_llm = self.llm.with_structured_output(LegalAnswerJudgment)
            judgment = structured_llm.invoke([HumanMessage(content=prompt)])
            
            # Ensure we have a proper judgment object
            if not isinstance(judgment, LegalAnswerJudgment):
                # Fallback to manual parsing if needed
                judgment = self._parse_unstructured_response(str(judgment))
                
        except Exception as e:
            print(f"Warning: LLM judge evaluation failed: {e}")
            # Return a default judgment
            judgment = self._create_default_judgment()
        
        return judgment
    
    def _parse_unstructured_response(self, response: str) -> LegalAnswerJudgment:
        """
        Parse an unstructured response into LegalAnswerJudgment.
        Fallback method if structured output fails.
        """
        # Simple parsing - look for keywords
        def extract_level(text: str, metric: str) -> str:
            text_lower = text.lower()
            if "high" in text_lower:
                return "High"
            elif "medium" in text_lower:
                return "Medium"
            else:
                return "Low"
        
        # Create default assessments
        return LegalAnswerJudgment(
            factual_accuracy=MetricAssessment(
                level=extract_level(response, "factual"),
                justification="Unable to parse detailed justification"
            ),
            evidence_grounding=MetricAssessment(
                level=extract_level(response, "evidence"),
                justification="Unable to parse detailed justification"
            ),
            clarity_reasoning=MetricAssessment(
                level=extract_level(response, "clarity"),
                justification="Unable to parse detailed justification"
            ),
            uncertainty_awareness=MetricAssessment(
                level=extract_level(response, "uncertainty"),
                justification="Unable to parse detailed justification"
            ),
            overall_usefulness=MetricAssessment(
                level=extract_level(response, "usefulness"),
                justification="Unable to parse detailed justification"
            ),
            summary="Evaluation parsing failed - manual review recommended",
            strengths=["Unable to extract strengths"],
            weaknesses=["Unable to extract weaknesses"]
        )
    
    def _create_default_judgment(self) -> LegalAnswerJudgment:
        """Create a default judgment when evaluation fails."""
        default_assessment = MetricAssessment(
            level="Medium",
            justification="Automated evaluation unavailable - manual review recommended"
        )
        
        return LegalAnswerJudgment(
            factual_accuracy=default_assessment,
            evidence_grounding=default_assessment,
            clarity_reasoning=default_assessment,
            uncertainty_awareness=default_assessment,
            overall_usefulness=default_assessment,
            summary="Automated evaluation failed - manual review recommended",
            strengths=["Evaluation unavailable"],
            weaknesses=["Evaluation unavailable"]
        )
    
    def format_judgment(self, judgment: LegalAnswerJudgment) -> str:
        """
        Format judgment into a readable report.
        
        Args:
            judgment: The LegalAnswerJudgment to format
            
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 60)
        report.append("LLM JUDGE EVALUATION")
        report.append("=" * 60)
        
        # Individual metrics
        report.append("\n📊 EVALUATION METRICS:")
        report.append("-" * 40)
        
        report.append("\n1. Factual Accuracy: " + judgment.factual_accuracy.level)
        report.append("   " + judgment.factual_accuracy.justification)
        
        report.append("\n2. Evidence Grounding: " + judgment.evidence_grounding.level)
        report.append("   " + judgment.evidence_grounding.justification)
        
        report.append("\n3. Clarity & Reasoning: " + judgment.clarity_reasoning.level)
        report.append("   " + judgment.clarity_reasoning.justification)
        
        report.append("\n4. Uncertainty Awareness: " + judgment.uncertainty_awareness.level)
        report.append("   " + judgment.uncertainty_awareness.justification)
        
        report.append("\n5. Overall Usefulness: " + judgment.overall_usefulness.level)
        report.append("   " + judgment.overall_usefulness.justification)
        
        # Summary
        if judgment.summary:
            report.append("\n📝 SUMMARY:")
            report.append("-" * 40)
            report.append(judgment.summary)
        
        # Strengths and weaknesses
        if judgment.strengths:
            report.append("\n✅ STRENGTHS:")
            for strength in judgment.strengths:
                report.append(f"• {strength}")
        
        if judgment.weaknesses:
            report.append("\n⚠️ AREAS FOR IMPROVEMENT:")
            for weakness in judgment.weaknesses:
                report.append(f"• {weakness}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_judgment(self, judgment: LegalAnswerJudgment, output_file: str = None) -> str:
        """
        Save judgment to a JSON file.
        
        Args:
            judgment: The judgment to save
            output_file: Optional output file name
            
        Returns:
            Path to saved file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"llm_judgment_{timestamp}.json"
        
        # Convert to dictionary
        judgment_dict = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'factual_accuracy': {
                    'level': judgment.factual_accuracy.level,
                    'justification': judgment.factual_accuracy.justification
                },
                'evidence_grounding': {
                    'level': judgment.evidence_grounding.level,
                    'justification': judgment.evidence_grounding.justification
                },
                'clarity_reasoning': {
                    'level': judgment.clarity_reasoning.level,
                    'justification': judgment.clarity_reasoning.justification
                },
                'uncertainty_awareness': {
                    'level': judgment.uncertainty_awareness.level,
                    'justification': judgment.uncertainty_awareness.justification
                },
                'overall_usefulness': {
                    'level': judgment.overall_usefulness.level,
                    'justification': judgment.overall_usefulness.justification
                }
            },
            'summary': judgment.summary,
            'strengths': judgment.strengths,
            'weaknesses': judgment.weaknesses
        }
        
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(judgment_dict, f, indent=2)
        
        return str(output_path)


class CombinedEvaluator:
    """
    Combined evaluator that provides both quantitative (U-score) and qualitative (LLM judge) evaluation.
    """
    
    def __init__(self, llm_model: str = "openai:gpt-4o"):
        """
        Initialize combined evaluator.
        
        Args:
            llm_model: Model to use for LLM judge
        """
        from .evaluation import LegalAnswerEvaluator
        self.quantitative_evaluator = LegalAnswerEvaluator()
        self.qualitative_evaluator = LLMJudgeEvaluator(llm_model)
    
    def evaluate(self, 
                 answer_text: str,
                 sources: List[str] = None,
                 jurisdiction: str = None,
                 question: str = None,
                 search_results: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform combined evaluation.
        
        Args:
            answer_text: The answer to evaluate
            sources: List of sources cited
            jurisdiction: Expected jurisdiction
            question: Original question (required for LLM judge)
            search_results: Search results used
            
        Returns:
            Dictionary containing both evaluations
        """
        # Quantitative evaluation (U-score)
        quantitative_metrics = self.quantitative_evaluator.evaluate(
            answer_text, sources, jurisdiction
        )
        
        # Qualitative evaluation (LLM judge)
        qualitative_judgment = None
        if question:
            qualitative_judgment = self.qualitative_evaluator.evaluate(
                question, answer_text, sources, search_results
            )
        
        return {
            'quantitative': quantitative_metrics,
            'qualitative': qualitative_judgment,
            'combined_assessment': self._generate_combined_assessment(
                quantitative_metrics, qualitative_judgment
            )
        }
    
    def _generate_combined_assessment(self, quant, qual) -> str:
        """Generate a combined assessment from both evaluations."""
        assessment = []
        
        # U-score interpretation
        if quant.u_score < 0.3:
            u_level = "Low uncertainty"
        elif quant.u_score < 0.6:
            u_level = "Moderate uncertainty"
        else:
            u_level = "High uncertainty"
        
        assessment.append(f"Quantitative: U-score {quant.u_score:.3f} ({u_level})")
        
        if qual:
            assessment.append(f"Qualitative: Overall {qual.overall_usefulness.level} usefulness")
            
            # Check for alignment
            quant_good = quant.u_score < 0.4
            qual_good = qual.overall_usefulness.level == "High"
            
            if quant_good and qual_good:
                assessment.append("✅ Both evaluations indicate high quality")
            elif not quant_good and not qual_good:
                assessment.append("⚠️ Both evaluations indicate quality concerns")
            else:
                assessment.append("🔍 Mixed evaluation results - manual review recommended")
        
        return " | ".join(assessment)
    
    def format_combined_report(self, evaluation: Dict[str, Any]) -> str:
        """Format a combined evaluation report."""
        report = []
        report.append("=" * 70)
        report.append("COMBINED EVALUATION REPORT")
        report.append("=" * 70)
        
        # Combined assessment
        report.append("\n🎯 COMBINED ASSESSMENT:")
        report.append(evaluation['combined_assessment'])
        
        # Quantitative section
        report.append("\n" + "=" * 70)
        report.append("QUANTITATIVE EVALUATION (U-SCORE)")
        report.append("=" * 70)
        quant = evaluation['quantitative']
        report.append(self.quantitative_evaluator.format_report(quant))
        
        # Qualitative section
        if evaluation['qualitative']:
            report.append("\n" + "=" * 70)
            report.append("QUALITATIVE EVALUATION (LLM JUDGE)")
            report.append("=" * 70)
            qual = evaluation['qualitative']
            report.append(self.qualitative_evaluator.format_judgment(qual))
        
        return "\n".join(report)