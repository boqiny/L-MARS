"""
Evaluation Module for L-MARS Legal Answers
Computes Uncertainty Score (U-score) based on multiple quality metrics
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

@dataclass
class EvaluationMetrics:
    """Detailed evaluation metrics for legal answer quality."""
    hedging_score: float  # H: Hedging cue rate (0=no hedges, 1=very hedgy)
    temporal_vagueness: float  # T: Temporal vagueness (0=precise, 1=vague)
    citation_score: float  # C: Citation sufficiency (0=strong citations, 1=weak/no citations)
    jurisdiction_score: float  # J: Jurisdiction specificity (0=clear, 1=missing)
    decisiveness_score: float  # M: Decisiveness of conclusion (0=decisive, 1=non-committal)
    u_score: float  # Overall uncertainty score (0=certain, 1=uncertain)
    
    # Detailed breakdowns
    hedging_details: Dict[str, Any]
    temporal_details: Dict[str, Any]
    citation_details: Dict[str, Any]
    jurisdiction_details: Dict[str, Any]
    decisiveness_details: Dict[str, Any]


class LegalAnswerEvaluator:
    """Evaluator for legal answer quality and uncertainty."""
    
    # Hedging lexicon for epistemic uncertainty
    HEDGING_CUES = {
        'epistemic': [
            'may', 'might', 'could', 'possibly', 'potentially', 'likely',
            'probably', 'perhaps', 'seemingly', 'apparently', 'arguably',
            'presumably', 'conceivably', 'plausibly'
        ],
        'perception': [
            'seems', 'appears', 'looks like', 'suggests', 'indicates',
            'implies', 'hints', 'it seems', 'it appears'
        ],
        'uncertainty': [
            'unclear', 'uncertain', 'unsure', 'ambiguous', 'debatable',
            'questionable', 'doubtful', 'unknown', 'undetermined'
        ],
        'qualification': [
            'generally', 'typically', 'usually', 'often', 'sometimes',
            'occasionally', 'rarely', 'seldom', 'in some cases', 'in certain cases'
        ],
        'assumption': [
            'assume', 'assuming', 'suppose', 'supposing', 'hypothetically',
            'theoretically', 'if', 'provided that', 'in case'
        ],
        'opinion': [
            'I think', 'I believe', 'in my opinion', 'in my view',
            'to my knowledge', 'as far as I know', 'from what I understand'
        ],
        'temporal_uncertainty': [
            'as of my last update', 'subject to change', 'may have changed',
            'at the time of writing', 'current as of', 'last I checked'
        ]
    }
    
    # Temporal vagueness patterns
    VAGUE_TEMPORAL = [
        'recent', 'recently', 'lately', 'current', 'currently',
        'ongoing', 'pending', 'forthcoming', 'upcoming', 'soon',
        'in the near future', 'in the past', 'historically',
        'as of my last update', 'at some point', 'eventually'
    ]
    
    # Concrete temporal patterns (dates, years, specific times)
    CONCRETE_TEMPORAL = re.compile(
        r'\b(?:\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|'
        r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|'
        r'(?:Q[1-4]\s+)?\d{4}|'
        r'(?:Spring|Summer|Fall|Winter)\s+\d{4})\b',
        re.IGNORECASE
    )
    
    # Authority weights for different citation sources
    CITATION_AUTHORITY = {
        'primary': {
            'patterns': [r'\.gov', r'\.court', r'supreme\.justia', r'courtlistener\.com', 
                        r'\.eu/|\.europa\.eu', r'gdpr-info\.eu', r'eur-lex\.europa\.eu'],
            'weight': 1.0
        },
        'official': {
            'patterns': [r'\.edu', r'justia\.com', r'findlaw\.com', r'law\.cornell', 
                        r'\.org\.uk', r'\.ac\.uk'],
            'weight': 0.7
        },
        'reputable': {
            'patterns': [r'nolo\.com', r'avvo\.com', r'lawyers\.com', r'americanbar\.org',
                        r'lexology\.com', r'mondaq\.com'],
            'weight': 0.5
        },
        'user_generated': {
            'patterns': [r'reddit\.com', r'quora\.com', r'forum', r'blog', r'medium\.com'],
            'weight': 0.2
        }
    }
    
    # Non-committal conclusion patterns
    NON_COMMITTAL_PATTERNS = [
        'consult.*(?:attorney|lawyer|counsel)',
        'seek.*(?:legal|professional).*advice',
        'situation.*(?:could|may|might).*(?:evolve|change|vary)',
        'depends on.*(?:specific|individual|particular).*circumstances',
        'cannot provide.*(?:legal|specific).*advice',
        'for.*(?:informational|educational).*purposes',
        'not.*legal.*advice',
        'varies.*by.*(?:state|jurisdiction|location)',
        'case.*by.*case.*basis'
    ]
    
    def __init__(self):
        """Initialize the evaluator."""
        self.all_hedges = []
        for category_hedges in self.HEDGING_CUES.values():
            self.all_hedges.extend(category_hedges)
    
    def evaluate(self, answer_text: str, sources: List[str] = None, 
                 jurisdiction_context: str = None) -> EvaluationMetrics:
        """
        Evaluate a legal answer for uncertainty and quality.
        
        Args:
            answer_text: The legal answer text to evaluate
            sources: List of source URLs/citations used
            jurisdiction_context: Expected jurisdiction from the query
            
        Returns:
            EvaluationMetrics with detailed scoring
        """
        # Tokenize for analysis (simple word tokenization)
        tokens = answer_text.lower().split()
        token_count = len(tokens)
        
        # 1. Calculate Hedging Cue Rate (H)
        hedging_score, hedging_details = self._calculate_hedging_score(answer_text, tokens, token_count)
        
        # 2. Calculate Temporal Vagueness (T)
        temporal_score, temporal_details = self._calculate_temporal_vagueness(answer_text)
        
        # 3. Calculate Citation Sufficiency & Authority (C)
        citation_score, citation_details = self._calculate_citation_score(sources or [], answer_text)
        
        # 4. Calculate Jurisdiction Specificity (J)
        jurisdiction_score, jurisdiction_details = self._calculate_jurisdiction_score(
            answer_text, jurisdiction_context
        )
        
        # 5. Calculate Decisiveness of Conclusion (M)
        decisiveness_score, decisiveness_details = self._calculate_decisiveness_score(answer_text)
        
        # Calculate overall U-score (adjusted weights for better balance)
        u_score = (
            0.20 * hedging_score +      # Reduced from 0.25
            0.10 * temporal_score +      # Reduced from 0.15
            0.25 * citation_score +      # Reduced from 0.30
            0.20 * jurisdiction_score +  # Increased from 0.15
            0.25 * decisiveness_score    # Increased from 0.15
        )
        
        return EvaluationMetrics(
            hedging_score=hedging_score,
            temporal_vagueness=temporal_score,
            citation_score=citation_score,
            jurisdiction_score=jurisdiction_score,
            decisiveness_score=decisiveness_score,
            u_score=u_score,
            hedging_details=hedging_details,
            temporal_details=temporal_details,
            citation_details=citation_details,
            jurisdiction_details=jurisdiction_details,
            decisiveness_details=decisiveness_details
        )
    
    def _calculate_hedging_score(self, text: str, tokens: List[str], token_count: int) -> Tuple[float, Dict]:
        """Calculate hedging cue rate (H)."""
        text_lower = text.lower()
        hedge_counts = {}
        total_hedges = 0
        
        for category, hedges in self.HEDGING_CUES.items():
            category_count = 0
            found_hedges = []
            for hedge in hedges:
                # Use word boundary regex for accurate matching
                pattern = r'\b' + re.escape(hedge) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    category_count += matches
                    found_hedges.append(hedge)
            hedge_counts[category] = {
                'count': category_count,
                'hedges': found_hedges
            }
            total_hedges += category_count
        
        # Calculate hedges per 100 tokens, cap at 10
        if token_count > 0:
            hedges_per_100 = (total_hedges / token_count) * 100
            hedges_per_100_capped = min(hedges_per_100, 10)
            score = hedges_per_100_capped / 10
        else:
            score = 0
            hedges_per_100 = 0
        
        details = {
            'total_hedges': total_hedges,
            'token_count': token_count,
            'hedges_per_100_tokens': hedges_per_100,
            'categories': hedge_counts,
            'score_interpretation': self._interpret_hedging_score(score)
        }
        
        return score, details
    
    def _calculate_temporal_vagueness(self, text: str) -> Tuple[float, Dict]:
        """Calculate temporal vagueness (T)."""
        text_lower = text.lower()
        
        # Find vague temporal mentions
        vague_mentions = []
        for vague_term in self.VAGUE_TEMPORAL:
            pattern = r'\b' + re.escape(vague_term) + r'\b'
            if re.search(pattern, text_lower):
                vague_mentions.append(vague_term)
        
        # Find concrete temporal mentions
        concrete_mentions = self.CONCRETE_TEMPORAL.findall(text)
        
        total_temporal = len(vague_mentions) + len(concrete_mentions)
        
        if total_temporal > 0:
            score = len(vague_mentions) / total_temporal
        else:
            # No temporal mentions at all is somewhat vague
            score = 0.5
        
        details = {
            'vague_mentions': vague_mentions,
            'concrete_mentions': concrete_mentions,
            'total_temporal_references': total_temporal,
            'score_interpretation': self._interpret_temporal_score(score)
        }
        
        return score, details
    
    def _calculate_citation_score(self, sources: List[str], text: str) -> Tuple[float, Dict]:
        """Calculate citation sufficiency and authority (C)."""
        if not sources:
            # Check for inline citations in text
            url_pattern = r'https?://[^\s)]+'
            urls_in_text = re.findall(url_pattern, text)
            sources = urls_in_text
        
        if not sources:
            return 1.0, {
                'total_citations': 0,
                'authority_breakdown': {},
                'total_weight': 0,
                'score_interpretation': 'No citations provided - maximum uncertainty'
            }
        
        authority_breakdown = {}
        total_weight = 0
        
        for source in sources:
            source_lower = source.lower()
            assigned = False
            
            for auth_level, auth_info in self.CITATION_AUTHORITY.items():
                for pattern in auth_info['patterns']:
                    if re.search(pattern, source_lower):
                        if auth_level not in authority_breakdown:
                            authority_breakdown[auth_level] = {
                                'count': 0,
                                'sources': [],
                                'weight': auth_info['weight']
                            }
                        authority_breakdown[auth_level]['count'] += 1
                        authority_breakdown[auth_level]['sources'].append(source[:50])
                        total_weight += auth_info['weight']
                        assigned = True
                        break
                if assigned:
                    break
            
            if not assigned:
                # Unknown source, assign minimal weight
                if 'unknown' not in authority_breakdown:
                    authority_breakdown['unknown'] = {
                        'count': 0,
                        'sources': [],
                        'weight': 0.1
                    }
                authority_breakdown['unknown']['count'] += 1
                authority_breakdown['unknown']['sources'].append(source[:50])
                total_weight += 0.1
        
        # Target weight is 1.5 (one primary or two secondary sources)
        target_weight = 1.5
        score = 1 - min(1, total_weight / target_weight)
        
        details = {
            'total_citations': len(sources),
            'authority_breakdown': authority_breakdown,
            'total_weight': total_weight,
            'target_weight': target_weight,
            'score_interpretation': self._interpret_citation_score(score)
        }
        
        return score, details
    
    def _calculate_jurisdiction_score(self, text: str, expected_jurisdiction: str = None) -> Tuple[float, Dict]:
        """Calculate jurisdiction specificity (J)."""
        # Common jurisdiction patterns
        jurisdiction_patterns = {
            'federal': r'\b(?:federal|U\.?S\.?|United States|nationwide)\b',
            'state': r'\b(?:state|California|New York|Texas|Florida|Illinois|Pennsylvania|Ohio)\b',
            'eu': r'\b(?:EU|European Union|Europe|GDPR|General Data Protection Regulation)\b',
            'specific_court': r'\b(?:Supreme Court|Circuit|District Court|Court of Appeals)\b',
            'location': r'\b(?:jurisdiction|county|city|municipality|local)\b'
        }
        
        found_jurisdictions = {}
        text_lower = text.lower()
        
        for jtype, pattern in jurisdiction_patterns.items():
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                found_jurisdictions[jtype] = matches
        
        # Scoring logic
        if not found_jurisdictions:
            score = 1.0  # No jurisdiction mentioned
            interpretation = "No jurisdiction specified - maximum uncertainty"
        elif expected_jurisdiction:
            # Check if expected jurisdiction is mentioned
            if expected_jurisdiction.lower() in text_lower:
                score = 0.0  # Correctly specified
                interpretation = f"Correctly specified jurisdiction: {expected_jurisdiction}"
            else:
                score = 0.5  # Mentioned but possibly mismatched
                interpretation = "Jurisdiction mentioned but may not match query context"
        else:
            # No expected jurisdiction, but some jurisdiction mentioned
            if len(found_jurisdictions) >= 2:
                score = 0.2  # Multiple jurisdictions mentioned
                interpretation = "Multiple jurisdictions clearly distinguished"
            else:
                score = 0.4  # Single jurisdiction mentioned
                interpretation = "Jurisdiction mentioned but not comprehensive"
        
        details = {
            'found_jurisdictions': found_jurisdictions,
            'expected_jurisdiction': expected_jurisdiction,
            'total_jurisdiction_mentions': sum(len(v) for v in found_jurisdictions.values()),
            'score_interpretation': interpretation
        }
        
        return score, details
    
    def _calculate_decisiveness_score(self, text: str) -> Tuple[float, Dict]:
        """Calculate decisiveness of conclusion (M)."""
        text_lower = text.lower()
        
        # Check for non-committal patterns
        non_committal_found = []
        for pattern in self.NON_COMMITTAL_PATTERNS:
            if re.search(pattern, text_lower):
                non_committal_found.append(pattern)
        
        # Check for decisive language
        decisive_patterns = [
            r'\b(?:yes|no|definitely|certainly|absolutely|clearly)\b',
            r'\b(?:must|required|prohibited|allowed|permitted|forbidden)\b',
            r'\b(?:legal|illegal|lawful|unlawful|valid|invalid)\b',
            r'\b(?:entitled|eligible|qualified|authorized)\b',
            r'\b(?:up to|maximum|minimum|exactly|specifically)\b',
            r'\b(?:is determined|will be|can be|applies to)\b',
            r'\d+\s*(?:million|euros?|dollars?|%|percent)'  # Specific numbers
        ]
        
        decisive_found = []
        for pattern in decisive_patterns:
            if re.search(pattern, text_lower):
                decisive_found.append(pattern)
        
        # Check for conditional but clear conclusions
        conditional_patterns = [
            r'if.*then',
            r'provided that',
            r'as long as',
            r'when.*(?:you|the)',
            r'(?:yes|no).*(?:if|when|provided)'
        ]
        
        conditional_found = []
        for pattern in conditional_patterns:
            if re.search(pattern, text_lower):
                conditional_found.append(pattern)
        
        # Scoring logic
        if len(non_committal_found) >= 3:
            score = 1.0  # Very non-committal
            interpretation = "Highly non-committal - multiple hedging disclaimers"
        elif len(non_committal_found) >= 1 and len(decisive_found) == 0:
            score = 0.8  # Non-committal without decisive language
            interpretation = "Non-committal conclusion without clear guidance"
        elif len(conditional_found) >= 1 and len(decisive_found) >= 1:
            score = 0.3  # Conditional but clear
            interpretation = "Clear conclusion with appropriate conditions"
        elif len(decisive_found) >= 2:
            score = 0.0  # Very decisive
            interpretation = "Decisive conclusion with clear legal guidance"
        else:
            score = 0.5  # Moderate decisiveness
            interpretation = "Moderate decisiveness - some guidance provided"
        
        details = {
            'non_committal_patterns': non_committal_found[:5],  # Limit to 5
            'decisive_patterns': decisive_found[:5],
            'conditional_patterns': conditional_found[:5],
            'score_interpretation': interpretation
        }
        
        return score, details
    
    def _interpret_hedging_score(self, score: float) -> str:
        """Interpret hedging score."""
        if score < 0.2:
            return "Very confident - minimal hedging"
        elif score < 0.4:
            return "Confident - some appropriate hedging"
        elif score < 0.6:
            return "Moderate - balanced hedging"
        elif score < 0.8:
            return "Uncertain - significant hedging"
        else:
            return "Very uncertain - excessive hedging"
    
    def _interpret_temporal_score(self, score: float) -> str:
        """Interpret temporal vagueness score."""
        if score < 0.2:
            return "Very precise temporal references"
        elif score < 0.4:
            return "Mostly precise with some vagueness"
        elif score < 0.6:
            return "Balanced precision and vagueness"
        elif score < 0.8:
            return "Mostly vague temporal references"
        else:
            return "Very vague or no temporal specificity"
    
    def _interpret_citation_score(self, score: float) -> str:
        """Interpret citation score."""
        if score < 0.2:
            return "Excellent - strong primary sources"
        elif score < 0.4:
            return "Good - adequate authoritative sources"
        elif score < 0.6:
            return "Fair - some authoritative sources"
        elif score < 0.8:
            return "Weak - mostly secondary sources"
        else:
            return "Very weak - no or only user-generated sources"
    
    def format_report(self, metrics: EvaluationMetrics) -> str:
        """Format evaluation metrics into a readable report."""
        report = []
        report.append("=" * 60)
        report.append("LEGAL ANSWER EVALUATION REPORT")
        report.append("=" * 60)
        
        # Overall score
        report.append(f"\n📊 OVERALL UNCERTAINTY SCORE (U-Score): {metrics.u_score:.3f}")
        report.append(f"   Interpretation: {self._interpret_u_score(metrics.u_score)}")
        
        # Component scores
        report.append("\n📈 COMPONENT SCORES:")
        report.append("-" * 40)
        
        # Hedging
        report.append(f"\n1. Hedging Cue Rate (H): {metrics.hedging_score:.3f}")
        report.append(f"   Weight: 20% | {metrics.hedging_details['score_interpretation']}")
        report.append(f"   Total hedges found: {metrics.hedging_details['total_hedges']}")
        if metrics.hedging_details['total_hedges'] > 0:
            for category, info in metrics.hedging_details['categories'].items():
                if info['count'] > 0:
                    report.append(f"   • {category}: {info['count']} ({', '.join(info['hedges'][:3])}...)")
        
        # Temporal
        report.append(f"\n2. Temporal Vagueness (T): {metrics.temporal_vagueness:.3f}")
        report.append(f"   Weight: 10% | {metrics.temporal_details['score_interpretation']}")
        if metrics.temporal_details['vague_mentions']:
            report.append(f"   Vague: {', '.join(metrics.temporal_details['vague_mentions'][:3])}")
        if metrics.temporal_details['concrete_mentions']:
            report.append(f"   Concrete: {', '.join(metrics.temporal_details['concrete_mentions'][:3])}")
        
        # Citations
        report.append(f"\n3. Citation Authority (C): {metrics.citation_score:.3f}")
        report.append(f"   Weight: 25% | {metrics.citation_details['score_interpretation']}")
        report.append(f"   Total citations: {metrics.citation_details['total_citations']}")
        report.append(f"   Authority weight: {metrics.citation_details['total_weight']:.2f}/{metrics.citation_details.get('target_weight', 1.5):.1f}")
        
        # Jurisdiction
        report.append(f"\n4. Jurisdiction Specificity (J): {metrics.jurisdiction_score:.3f}")
        report.append(f"   Weight: 20% | {metrics.jurisdiction_details['score_interpretation']}")
        if metrics.jurisdiction_details['found_jurisdictions']:
            report.append(f"   Found: {', '.join(metrics.jurisdiction_details['found_jurisdictions'].keys())}")
        
        # Decisiveness
        report.append(f"\n5. Decisiveness (M): {metrics.decisiveness_score:.3f}")
        report.append(f"   Weight: 25% | {metrics.decisiveness_details['score_interpretation']}")
        
        # Recommendations
        report.append("\n💡 RECOMMENDATIONS:")
        report.append("-" * 40)
        recommendations = self._generate_recommendations(metrics)
        for rec in recommendations:
            report.append(f"• {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def _interpret_u_score(self, score: float) -> str:
        """Interpret overall U-score."""
        if score < 0.2:
            return "✅ Highly certain and authoritative answer"
        elif score < 0.4:
            return "✅ Good certainty with minor uncertainties"
        elif score < 0.6:
            return "⚠️ Moderate uncertainty - use with caution"
        elif score < 0.8:
            return "⚠️ High uncertainty - significant limitations"
        else:
            return "❌ Very high uncertainty - unreliable for decision-making"
    
    def _generate_recommendations(self, metrics: EvaluationMetrics) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if metrics.hedging_score > 0.6:
            recommendations.append("Reduce hedging language for more authoritative guidance")
        
        if metrics.temporal_vagueness > 0.6:
            recommendations.append("Add specific dates and temporal references")
        
        if metrics.citation_score > 0.6:
            recommendations.append("Add primary legal sources (government, court decisions)")
        
        if metrics.jurisdiction_score > 0.6:
            recommendations.append("Clearly specify applicable jurisdiction(s)")
        
        if metrics.decisiveness_score > 0.6:
            recommendations.append("Provide clearer conclusions with specific conditions")
        
        if not recommendations:
            recommendations.append("Answer meets quality standards - maintain current approach")
        
        return recommendations
    
    def save_evaluation(self, metrics: EvaluationMetrics, output_file: str = None) -> str:
        """Save evaluation metrics to a JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_{timestamp}.json"
        
        # Convert to dictionary
        eval_dict = {
            'timestamp': datetime.now().isoformat(),
            'u_score': metrics.u_score,
            'component_scores': {
                'hedging': metrics.hedging_score,
                'temporal': metrics.temporal_vagueness,
                'citation': metrics.citation_score,
                'jurisdiction': metrics.jurisdiction_score,
                'decisiveness': metrics.decisiveness_score
            },
            'details': {
                'hedging': metrics.hedging_details,
                'temporal': metrics.temporal_details,
                'citation': metrics.citation_details,
                'jurisdiction': metrics.jurisdiction_details,
                'decisiveness': metrics.decisiveness_details
            }
        }
        
        # Ensure results directory exists
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        output_path = results_dir / output_file
        with open(output_path, 'w') as f:
            json.dump(eval_dict, f, indent=2)
        
        return str(output_path)


def evaluate_answer(answer_text: str, sources: List[str] = None, 
                   jurisdiction: str = None, save_results: bool = False) -> EvaluationMetrics:
    """
    Convenience function to evaluate a legal answer.
    
    Args:
        answer_text: The legal answer to evaluate
        sources: List of sources/citations
        jurisdiction: Expected jurisdiction
        save_results: Whether to save results to file
    
    Returns:
        EvaluationMetrics object
    """
    evaluator = LegalAnswerEvaluator()
    metrics = evaluator.evaluate(answer_text, sources, jurisdiction)
    
    if save_results:
        output_file = evaluator.save_evaluation(metrics)
        print(f"Evaluation saved to: {output_file}")
    
    return metrics


def evaluate_from_log(log_file: str, save_results: bool = True) -> Optional[EvaluationMetrics]:
    """
    Evaluate an answer from a saved log file.
    
    Args:
        log_file: Path to the L-MARS log file
        save_results: Whether to save evaluation results
    
    Returns:
        EvaluationMetrics object or None if no answer found
    """
    with open(log_file, 'r') as f:
        log_data = json.load(f)
    
    # Extract final answer
    final_answer = log_data.get('final_answer')
    if not final_answer:
        print("No final answer found in log file")
        return None
    
    answer_text = final_answer.get('answer', '')
    sources = final_answer.get('sources', [])
    
    # Try to extract jurisdiction from query or configuration
    query = log_data.get('query', '')
    jurisdiction = None
    
    # Simple jurisdiction extraction from query
    if 'california' in query.lower():
        jurisdiction = 'California'
    elif 'new york' in query.lower():
        jurisdiction = 'New York'
    elif 'federal' in query.lower() or 'us' in query.lower():
        jurisdiction = 'Federal'
    
    # Evaluate
    evaluator = LegalAnswerEvaluator()
    metrics = evaluator.evaluate(answer_text, sources, jurisdiction)
    
    # Print report
    print(evaluator.format_report(metrics))
    
    if save_results:
        # Save with matching session ID
        session_id = log_data.get('session_id', 'unknown')
        output_file = f"evaluation_{session_id}.json"
        output_path = evaluator.save_evaluation(metrics, output_file)
        print(f"\nEvaluation saved to: {output_path}")
    
    return metrics