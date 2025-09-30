#!/usr/bin/env python3
"""
Cross-Paper Synthesis Engine for KnowHunt

This module implements sophisticated analysis across multiple research papers
to identify trends, correlations, and emerging themes across different domains.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
import re
import statistics

# Import existing components
from minimal_pipeline import MinimalArXivCollector, MinimalLLMAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ResearchPaper:
    """Represents a research paper with analysis results."""
    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    published_date: str
    arxiv_id: Optional[str]
    url: str
    domain: str  # Inferred research domain
    keywords: List[str]
    analysis_result: Optional[Dict[str, Any]] = None


@dataclass
class CrossPaperTrend:
    """Represents a trend identified across multiple papers."""
    trend_id: str
    trend_name: str
    description: str
    supporting_papers: List[str]
    confidence_score: float
    trend_type: str  # 'technical', 'methodological', 'application', 'business'
    emergence_timeframe: str
    related_keywords: List[str]
    impact_assessment: Dict[str, Any]


@dataclass
class SynthesisResult:
    """Complete result of cross-paper synthesis analysis."""
    synthesis_id: str
    query_domains: List[str]
    papers_analyzed: int
    synthesis_timestamp: datetime

    # Core synthesis results
    cross_domain_trends: List[CrossPaperTrend]
    common_techniques: List[Dict[str, Any]]
    emerging_themes: List[Dict[str, Any]]
    research_gaps: List[str]
    collaboration_patterns: List[Dict[str, Any]]
    methodological_convergence: List[Dict[str, Any]]

    # Meta analysis
    synthesis_quality_score: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    limitations: List[str]
    recommendations: List[str]


class DomainClassifier:
    """Classifies research papers into domains based on content."""

    DOMAIN_KEYWORDS = {
        'computer_vision': ['vision', 'image', 'visual', 'cnn', 'convolution', 'detection', 'segmentation', 'recognition'],
        'nlp': ['language', 'text', 'linguistic', 'nlp', 'bert', 'transformer', 'tokeniz', 'semantic'],
        'ml_theory': ['learning theory', 'optimization', 'convergence', 'generalization', 'statistical learning'],
        'reinforcement_learning': ['reinforcement', 'rl', 'agent', 'environment', 'reward', 'policy', 'q-learning'],
        'deep_learning': ['deep', 'neural network', 'backpropagation', 'gradient descent', 'activation'],
        'ai_safety': ['safety', 'alignment', 'robustness', 'interpretability', 'explainable', 'fairness'],
        'robotics': ['robot', 'manipulation', 'locomotion', 'control', 'autonomous', 'sensors'],
        'quantum_computing': ['quantum', 'qubit', 'entanglement', 'superposition', 'gate'],
        'bioinformatics': ['protein', 'dna', 'gene', 'biological', 'molecular', 'genomic'],
        'healthcare_ai': ['medical', 'clinical', 'diagnosis', 'healthcare', 'patient', 'disease']
    }

    def classify_paper(self, paper: ResearchPaper) -> str:
        """Classify a paper into a research domain."""
        text = f"{paper.title} {paper.abstract}".lower()

        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return 'general_ai'


class TrendDetector:
    """Detects trends across multiple research papers."""

    def __init__(self, min_papers_for_trend: int = 3, confidence_threshold: float = 0.6):
        self.min_papers_for_trend = min_papers_for_trend
        self.confidence_threshold = confidence_threshold

    def detect_technical_trends(self, papers: List[ResearchPaper]) -> List[CrossPaperTrend]:
        """Detect technical trends across papers."""
        trends = []

        # Analyze technique mentions
        technique_counts = Counter()
        technique_papers = defaultdict(set)

        for paper in papers:
            if paper.analysis_result and 'technical_innovations' in paper.analysis_result:
                for innovation in paper.analysis_result['technical_innovations']:
                    # Extract key technical terms
                    tech_terms = self._extract_technical_terms(innovation)
                    for term in tech_terms:
                        technique_counts[term] += 1
                        technique_papers[term].add(paper.paper_id)

        # Find trends (techniques appearing in multiple papers)
        for technique, count in technique_counts.items():
            if count >= self.min_papers_for_trend:
                supporting_papers = list(technique_papers[technique])
                confidence = min(count / len(papers), 1.0)

                if confidence >= self.confidence_threshold:
                    trend = CrossPaperTrend(
                        trend_id=f"tech_trend_{hashlib.md5(technique.encode()).hexdigest()[:8]}",
                        trend_name=technique,
                        description=f"Technical trend involving {technique} across {count} papers",
                        supporting_papers=supporting_papers,
                        confidence_score=confidence,
                        trend_type='technical',
                        emergence_timeframe=self._estimate_emergence_timeframe(papers, supporting_papers),
                        related_keywords=[technique],
                        impact_assessment={'adoption_rate': count / len(papers)}
                    )
                    trends.append(trend)

        return trends

    def detect_methodological_trends(self, papers: List[ResearchPaper]) -> List[CrossPaperTrend]:
        """Detect methodological trends across papers."""
        trends = []

        # Common methodological patterns
        method_patterns = {
            'self_supervised': ['self-supervised', 'unsupervised pre-training', 'contrastive learning'],
            'attention_mechanisms': ['attention', 'transformer', 'self-attention'],
            'meta_learning': ['meta-learning', 'few-shot', 'learning to learn'],
            'adversarial_training': ['adversarial', 'gan', 'generative adversarial'],
            'transfer_learning': ['transfer', 'fine-tuning', 'pre-trained'],
            'multimodal_fusion': ['multimodal', 'cross-modal', 'fusion']
        }

        method_counts = defaultdict(int)
        method_papers = defaultdict(set)

        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            if paper.analysis_result:
                text += " " + " ".join(paper.analysis_result.get('technical_innovations', []))

            for method_name, patterns in method_patterns.items():
                if any(pattern in text for pattern in patterns):
                    method_counts[method_name] += 1
                    method_papers[method_name].add(paper.paper_id)

        for method, count in method_counts.items():
            if count >= self.min_papers_for_trend:
                confidence = count / len(papers)
                if confidence >= self.confidence_threshold:
                    trend = CrossPaperTrend(
                        trend_id=f"method_trend_{method}",
                        trend_name=method.replace('_', ' ').title(),
                        description=f"Methodological trend in {method.replace('_', ' ')} appearing in {count} papers",
                        supporting_papers=list(method_papers[method]),
                        confidence_score=confidence,
                        trend_type='methodological',
                        emergence_timeframe=self._estimate_emergence_timeframe(papers, method_papers[method]),
                        related_keywords=method_patterns[method],
                        impact_assessment={'prevalence': confidence}
                    )
                    trends.append(trend)

        return trends

    def detect_application_trends(self, papers: List[ResearchPaper]) -> List[CrossPaperTrend]:
        """Detect application domain trends."""
        trends = []

        # Application domains
        app_domains = {
            'healthcare': ['medical', 'clinical', 'healthcare', 'diagnosis', 'treatment'],
            'autonomous_systems': ['autonomous', 'self-driving', 'robotic', 'navigation'],
            'edge_computing': ['edge', 'mobile', 'embedded', 'iot', 'resource-constrained'],
            'sustainability': ['sustainable', 'energy-efficient', 'green', 'carbon', 'environmental'],
            'human_ai_interaction': ['human-ai', 'interaction', 'collaborative', 'interface', 'usability']
        }

        app_counts = defaultdict(int)
        app_papers = defaultdict(set)

        for paper in papers:
            text = f"{paper.title} {paper.abstract}".lower()
            if paper.analysis_result and 'business_implications' in paper.analysis_result:
                text += " " + " ".join(paper.analysis_result['business_implications'])

            for app_name, keywords in app_domains.items():
                if any(keyword in text for keyword in keywords):
                    app_counts[app_name] += 1
                    app_papers[app_name].add(paper.paper_id)

        for app, count in app_counts.items():
            if count >= self.min_papers_for_trend:
                confidence = count / len(papers)
                if confidence >= self.confidence_threshold:
                    trend = CrossPaperTrend(
                        trend_id=f"app_trend_{app}",
                        trend_name=app.replace('_', ' ').title(),
                        description=f"Application trend in {app.replace('_', ' ')} across {count} papers",
                        supporting_papers=list(app_papers[app]),
                        confidence_score=confidence,
                        trend_type='application',
                        emergence_timeframe=self._estimate_emergence_timeframe(papers, app_papers[app]),
                        related_keywords=app_domains[app],
                        impact_assessment={'market_relevance': confidence}
                    )
                    trends.append(trend)

        return trends

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from text."""
        # Simple technical term extraction
        terms = []

        # Look for common technical patterns
        patterns = [
            r'(\w+)\s+(network|architecture|algorithm|method|approach|technique)',
            r'(attention|transformer|convolution|recurrent|lstm|gru)',
            r'(\w+)\s+(learning|training|optimization)',
        ]

        text_lower = text.lower()
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if isinstance(match, tuple):
                    terms.extend([t for t in match if len(t) > 2])
                else:
                    terms.append(match)

        return list(set(terms))

    def _estimate_emergence_timeframe(self, papers: List[ResearchPaper], supporting_paper_ids: Set[str]) -> str:
        """Estimate when a trend emerged based on paper dates."""
        supporting_papers = [p for p in papers if p.paper_id in supporting_paper_ids]

        if not supporting_papers:
            return "unknown"

        # Simple heuristic based on paper count over time
        if len(supporting_papers) >= len(papers) * 0.5:
            return "established"
        elif len(supporting_papers) >= len(papers) * 0.3:
            return "growing"
        else:
            return "emerging"


class CrossPaperSynthesizer:
    """Main synthesizer for cross-paper analysis."""

    def __init__(self, model: str = "qwen2.5-coder:3b"):
        self.model = model
        self.domain_classifier = DomainClassifier()
        self.trend_detector = TrendDetector()
        self.llm_analyzer = MinimalLLMAnalyzer(model)

    async def synthesize_papers(self, papers: List[ResearchPaper]) -> SynthesisResult:
        """Perform comprehensive cross-paper synthesis."""
        logger.info(f"Starting synthesis of {len(papers)} papers")

        # Classify papers by domain
        for paper in papers:
            paper.domain = self.domain_classifier.classify_paper(paper)

        # Detect various types of trends
        technical_trends = self.trend_detector.detect_technical_trends(papers)
        methodological_trends = self.trend_detector.detect_methodological_trends(papers)
        application_trends = self.trend_detector.detect_application_trends(papers)

        all_trends = technical_trends + methodological_trends + application_trends

        # Generate common techniques analysis
        common_techniques = await self._analyze_common_techniques(papers)

        # Identify emerging themes
        emerging_themes = await self._identify_emerging_themes(papers, all_trends)

        # Detect research gaps
        research_gaps = await self._identify_research_gaps(papers)

        # Analyze collaboration patterns
        collaboration_patterns = self._analyze_collaboration_patterns(papers)

        # Assess methodological convergence
        methodological_convergence = self._analyze_methodological_convergence(papers)

        # Calculate synthesis quality
        quality_score = self._calculate_synthesis_quality(papers, all_trends)

        result = SynthesisResult(
            synthesis_id=f"synthesis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            query_domains=list(set(p.domain for p in papers)),
            papers_analyzed=len(papers),
            synthesis_timestamp=datetime.now(),
            cross_domain_trends=all_trends,
            common_techniques=common_techniques,
            emerging_themes=emerging_themes,
            research_gaps=research_gaps,
            collaboration_patterns=collaboration_patterns,
            methodological_convergence=methodological_convergence,
            synthesis_quality_score=quality_score,
            confidence_intervals=self._calculate_confidence_intervals(papers, all_trends),
            limitations=self._identify_limitations(papers),
            recommendations=await self._generate_recommendations(papers, all_trends)
        )

        logger.info(f"Synthesis complete: {len(all_trends)} trends, quality score {quality_score:.2f}")
        return result

    async def _analyze_common_techniques(self, papers: List[ResearchPaper]) -> List[Dict[str, Any]]:
        """Analyze techniques common across papers."""
        techniques = []

        # Extract technical innovations from all papers
        all_innovations = []
        for paper in papers:
            if paper.analysis_result and 'technical_innovations' in paper.analysis_result:
                all_innovations.extend(paper.analysis_result['technical_innovations'])

        if not all_innovations:
            return techniques

        # Use LLM to identify common patterns
        innovations_text = "\n".join([f"- {innovation}" for innovation in all_innovations])

        analysis_prompt = f"""
        Analyze these technical innovations from {len(papers)} research papers and identify common techniques or patterns:

        {innovations_text}

        Return JSON with common techniques:
        {{
            "common_techniques": [
                {{
                    "technique_name": "...",
                    "description": "...",
                    "frequency": 0.0,
                    "applications": ["..."]
                }}
            ]
        }}
        """

        try:
            analysis_result = await self.llm_analyzer.analyze(analysis_prompt, "Common Techniques Analysis")
            if analysis_result and 'technical_innovations' in analysis_result:
                # Parse the LLM response for common techniques
                techniques = [
                    {
                        "technique_name": f"Common Pattern {i+1}",
                        "description": innovation,
                        "frequency": 1.0 / len(all_innovations),
                        "applications": ["General AI Research"]
                    }
                    for i, innovation in enumerate(analysis_result['technical_innovations'][:5])
                ]
        except Exception as e:
            logger.warning(f"LLM analysis for common techniques failed: {e}")

        return techniques

    async def _identify_emerging_themes(self, papers: List[ResearchPaper], trends: List[CrossPaperTrend]) -> List[Dict[str, Any]]:
        """Identify emerging themes across papers."""
        themes = []

        # Group trends by type and confidence
        high_confidence_trends = [t for t in trends if t.confidence_score > 0.7]

        for trend in high_confidence_trends[:10]:  # Top 10 trends
            theme = {
                "theme_name": trend.trend_name,
                "description": trend.description,
                "confidence": trend.confidence_score,
                "supporting_evidence": len(trend.supporting_papers),
                "trend_type": trend.trend_type,
                "emergence_timeline": trend.emergence_timeframe
            }
            themes.append(theme)

        return themes

    async def _identify_research_gaps(self, papers: List[ResearchPaper]) -> List[str]:
        """Identify potential research gaps."""
        gaps = []

        # Domain coverage analysis
        domains = Counter(paper.domain for paper in papers)

        # Identify underrepresented domains
        if len(domains) > 1:
            min_coverage = min(domains.values())
            max_coverage = max(domains.values())

            if max_coverage / min_coverage > 3:  # Significant imbalance
                underrepresented = [domain for domain, count in domains.items() if count == min_coverage]
                gaps.append(f"Underrepresented research areas: {', '.join(underrepresented)}")

        # Methodological gaps
        methods_mentioned = set()
        for paper in papers:
            if paper.analysis_result and 'technical_innovations' in paper.analysis_result:
                for innovation in paper.analysis_result['technical_innovations']:
                    if 'evaluation' in innovation.lower() or 'benchmark' in innovation.lower():
                        methods_mentioned.add('evaluation')
                    if 'interpretabil' in innovation.lower() or 'explain' in innovation.lower():
                        methods_mentioned.add('interpretability')

        if 'evaluation' not in methods_mentioned:
            gaps.append("Limited focus on evaluation methodologies and benchmarking")

        if 'interpretability' not in methods_mentioned:
            gaps.append("Insufficient attention to model interpretability and explainability")

        return gaps

    def _analyze_collaboration_patterns(self, papers: List[ResearchPaper]) -> List[Dict[str, Any]]:
        """Analyze collaboration patterns between authors."""
        patterns = []

        # Simple collaboration analysis
        all_authors = []
        for paper in papers:
            all_authors.extend(paper.authors)

        author_counts = Counter(all_authors)
        prolific_authors = [author for author, count in author_counts.items() if count > 1]

        if prolific_authors:
            patterns.append({
                "pattern_type": "prolific_researchers",
                "description": f"{len(prolific_authors)} researchers appear in multiple papers",
                "researchers": prolific_authors[:5],  # Top 5
                "indicator": "high_productivity"
            })

        # Institution diversity (if available in author names)
        unique_authors = len(set(all_authors))
        total_authors = len(all_authors)

        if total_authors > 0:
            diversity_ratio = unique_authors / total_authors
            patterns.append({
                "pattern_type": "collaboration_diversity",
                "description": f"Author diversity ratio: {diversity_ratio:.2f}",
                "diversity_score": diversity_ratio,
                "indicator": "collaboration_breadth"
            })

        return patterns

    def _analyze_methodological_convergence(self, papers: List[ResearchPaper]) -> List[Dict[str, Any]]:
        """Analyze convergence in methodological approaches."""
        convergence = []

        # Domain-specific convergence
        domain_methods = defaultdict(list)

        for paper in papers:
            if paper.analysis_result and 'technical_innovations' in paper.analysis_result:
                domain_methods[paper.domain].extend(paper.analysis_result['technical_innovations'])

        for domain, methods in domain_methods.items():
            if len(methods) > 2:
                # Simple convergence indicator
                method_words = []
                for method in methods:
                    method_words.extend(method.lower().split())

                common_words = [word for word, count in Counter(method_words).items()
                              if count > 1 and len(word) > 3]

                if common_words:
                    convergence.append({
                        "domain": domain,
                        "convergence_type": "methodological",
                        "common_elements": common_words[:5],
                        "strength": len(common_words) / len(set(method_words))
                    })

        return convergence

    def _calculate_synthesis_quality(self, papers: List[ResearchPaper], trends: List[CrossPaperTrend]) -> float:
        """Calculate quality score for the synthesis."""
        factors = []

        # Factor 1: Number of papers
        papers_factor = min(len(papers) / 20, 1.0)  # Optimal around 20 papers
        factors.append(papers_factor)

        # Factor 2: Trend detection success
        trends_factor = min(len(trends) / 10, 1.0)  # Good synthesis has ~10 trends
        factors.append(trends_factor)

        # Factor 3: Domain diversity
        domains = set(paper.domain for paper in papers)
        diversity_factor = min(len(domains) / 5, 1.0)  # Good diversity is 5+ domains
        factors.append(diversity_factor)

        # Factor 4: Analysis completeness
        analyzed_papers = sum(1 for paper in papers if paper.analysis_result)
        completeness_factor = analyzed_papers / len(papers) if papers else 0
        factors.append(completeness_factor)

        return statistics.mean(factors)

    def _calculate_confidence_intervals(self, papers: List[ResearchPaper], trends: List[CrossPaperTrend]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics."""
        if not trends:
            return {}

        trend_confidences = [trend.confidence_score for trend in trends]

        mean_confidence = statistics.mean(trend_confidences)
        std_confidence = statistics.stdev(trend_confidences) if len(trend_confidences) > 1 else 0

        return {
            "trend_confidence": (
                max(0, mean_confidence - 1.96 * std_confidence),
                min(1, mean_confidence + 1.96 * std_confidence)
            ),
            "synthesis_reliability": (0.6, 0.9)  # Estimated based on methodology
        }

    def _identify_limitations(self, papers: List[ResearchPaper]) -> List[str]:
        """Identify limitations in the synthesis."""
        limitations = []

        if len(papers) < 10:
            limitations.append(f"Limited sample size: only {len(papers)} papers analyzed")

        # Check for LLM analysis availability
        analyzed_count = sum(1 for paper in papers if paper.analysis_result)
        if analyzed_count < len(papers) * 0.8:
            limitations.append(f"Incomplete LLM analysis: only {analyzed_count}/{len(papers)} papers fully analyzed")

        # Check domain balance
        domains = Counter(paper.domain for paper in papers)
        if len(domains) == 1:
            limitations.append("Single domain analysis - limited cross-domain insights")

        limitations.append("Synthesis based on abstracts and titles - full paper content not analyzed")
        limitations.append("Automated trend detection may miss nuanced patterns requiring expert knowledge")

        return limitations

    async def _generate_recommendations(self, papers: List[ResearchPaper], trends: List[CrossPaperTrend]) -> List[str]:
        """Generate actionable recommendations based on synthesis."""
        recommendations = []

        # Research direction recommendations
        high_confidence_trends = [t for t in trends if t.confidence_score > 0.7]

        if high_confidence_trends:
            top_trend = max(high_confidence_trends, key=lambda t: t.confidence_score)
            recommendations.append(f"Focus research efforts on {top_trend.trend_name} - showing strong cross-paper adoption")

        # Domain recommendations
        domains = Counter(paper.domain for paper in papers)
        if len(domains) > 1:
            dominant_domain = domains.most_common(1)[0][0]
            recommendations.append(f"Consider cross-domain applications from {dominant_domain} to other fields")

        # Methodological recommendations
        method_trends = [t for t in trends if t.trend_type == 'methodological']
        if method_trends:
            recommendations.append("Investigate methodological convergence patterns for standardization opportunities")

        # Gap-filling recommendations
        if len(papers) < 20:
            recommendations.append("Expand analysis to include more papers for better trend detection")

        recommendations.append("Conduct follow-up analysis with full paper texts for deeper insights")
        recommendations.append("Validate identified trends with domain experts")

        return recommendations


async def test_cross_paper_synthesis():
    """Test the cross-paper synthesis system."""
    print("🔬 Testing Cross-Paper Synthesis System")
    print("=" * 45)

    # Create test papers
    test_papers = [
        ResearchPaper(
            paper_id="paper_1",
            title="Attention Is All You Need: Transformer Architecture for Sequence Modeling",
            abstract="We propose a new architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.",
            authors=["Ashish Vaswani", "Noam Shazeer"],
            published_date="2017-06-12",
            arxiv_id="1706.03762",
            url="https://arxiv.org/abs/1706.03762",
            domain="nlp",
            keywords=["attention", "transformer", "sequence modeling"],
            analysis_result={
                "technical_innovations": ["self-attention mechanism", "positional encoding", "multi-head attention"],
                "business_implications": ["reduced training time", "better parallelization"],
                "research_quality_score": 9
            }
        ),
        ResearchPaper(
            paper_id="paper_2",
            title="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            abstract="We introduce BERT, a bidirectional transformer that achieves state-of-the-art results on eleven natural language processing tasks.",
            authors=["Jacob Devlin", "Ming-Wei Chang"],
            published_date="2018-10-11",
            arxiv_id="1810.04805",
            url="https://arxiv.org/abs/1810.04805",
            domain="nlp",
            keywords=["bert", "bidirectional", "transformer", "pre-training"],
            analysis_result={
                "technical_innovations": ["bidirectional training", "masked language modeling", "transformer encoder"],
                "business_implications": ["improved text understanding", "transfer learning capabilities"],
                "research_quality_score": 9
            }
        ),
        ResearchPaper(
            paper_id="paper_3",
            title="Vision Transformer: An Image is Worth 16x16 Words",
            abstract="We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.",
            authors=["Alexey Dosovitskiy", "Lucas Beyer"],
            published_date="2020-10-22",
            arxiv_id="2010.11929",
            url="https://arxiv.org/abs/2010.11929",
            domain="computer_vision",
            keywords=["vision transformer", "image patches", "attention"],
            analysis_result={
                "technical_innovations": ["patch embedding", "vision transformer", "attention for images"],
                "business_implications": ["unified architecture", "reduced domain-specific engineering"],
                "research_quality_score": 8
            }
        )
    ]

    # Test synthesis
    synthesizer = CrossPaperSynthesizer()
    result = await synthesizer.synthesize_papers(test_papers)

    # Display results
    print(f"✅ Synthesis completed for {result.papers_analyzed} papers")
    print(f"📊 Quality Score: {result.synthesis_quality_score:.2f}")
    print(f"🔍 Cross-domain trends found: {len(result.cross_domain_trends)}")

    print("\n🔥 Top Trends Detected:")
    for i, trend in enumerate(result.cross_domain_trends[:3], 1):
        print(f"  {i}. {trend.trend_name} ({trend.trend_type})")
        print(f"     Confidence: {trend.confidence_score:.2f}")
        print(f"     Papers: {len(trend.supporting_papers)}")

    print("\n🛠️ Common Techniques:")
    for i, technique in enumerate(result.common_techniques[:3], 1):
        print(f"  {i}. {technique.get('technique_name', 'Unknown')}")

    print("\n🌟 Emerging Themes:")
    for i, theme in enumerate(result.emerging_themes[:3], 1):
        print(f"  {i}. {theme.get('theme_name', 'Unknown')}")

    print("\n⚠️ Research Gaps:")
    for gap in result.research_gaps[:3]:
        print(f"  - {gap}")

    print("\n💡 Recommendations:")
    for rec in result.recommendations[:3]:
        print(f"  - {rec}")

    print("\n✅ Cross-paper synthesis test completed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run test
    asyncio.run(test_cross_paper_synthesis())