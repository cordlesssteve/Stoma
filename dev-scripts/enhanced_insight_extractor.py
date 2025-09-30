#!/usr/bin/env python3
"""
Enhanced Insight Extractor - Better analysis when LLM returns generic results
"""

import re
import json
from typing import List, Dict, Any


class InsightExtractor:
    """Extract meaningful insights from research papers when LLM fails."""

    def __init__(self):
        # Keywords that indicate important technical content
        self.technical_keywords = {
            'performance': ['performance', 'accuracy', 'efficiency', 'speed', 'latency', 'throughput'],
            'novel_methods': ['novel', 'new', 'first', 'introduces', 'proposes', 'breakthrough'],
            'improvements': ['improves', 'better', 'outperforms', 'reduces', 'increases', 'achieves'],
            'quantitative': [r'\d+%', r'\d+x', r'\d+\.\d+', 'times faster', 'reduction'],
        }

        self.business_keywords = [
            'cost', 'market', 'commercial', 'application', 'deployment', 'scalable',
            'enterprise', 'industry', 'practical', 'real-world', 'production'
        ]

    def enhance_analysis(self, analysis: Dict, papers: List[Dict]) -> Dict:
        """Enhance analysis with extracted insights when LLM returns generic results."""

        enhanced = analysis.copy()

        # Check if we have generic results (paper titles instead of insights)
        if self._has_generic_results(analysis, papers):
            print("🔧 Detected generic LLM results, enhancing with extracted insights...")

            enhanced['novel_contributions'] = self._extract_novel_contributions(papers)
            enhanced['technical_innovations'] = self._extract_technical_innovations(papers)
            enhanced['business_implications'] = self._extract_business_implications(papers)

        return enhanced

    def _has_generic_results(self, analysis: Dict, papers: List[Dict]) -> bool:
        """Check if analysis contains generic results (paper titles)."""
        paper_titles = [p.get('title', '').replace('\n', ' ').strip() for p in papers]
        contributions = analysis.get('novel_contributions', [])

        # Normalize titles and contributions for comparison
        normalized_titles = [title.lower().replace('  ', ' ') for title in paper_titles]
        normalized_contribs = [contrib.lower().replace('  ', ' ').strip() for contrib in contributions]

        # If most contributions match paper titles, it's generic
        generic_count = 0
        for contrib in normalized_contribs:
            for title in normalized_titles:
                if title in contrib or contrib in title:
                    generic_count += 1
                    break

        print(f"🔍 Generic detection: {generic_count}/{len(contributions)} contributions match titles")
        return generic_count >= 1  # Any title match means it's generic

    def _extract_novel_contributions(self, papers: List[Dict]) -> List[str]:
        """Extract actual novel contributions from paper abstracts."""
        contributions = []

        for paper in papers:
            abstract = paper.get('abstract', '')
            title = paper.get('title', '')

            # Look for novel/breakthrough language
            novel_sentences = []
            sentences = abstract.split('. ')

            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in self.technical_keywords['novel_methods']):
                    # Clean and format the contribution
                    contribution = self._clean_sentence(sentence)
                    if len(contribution) > 20:  # Ensure it's substantial
                        novel_sentences.append(f"{contribution}")

            # Add the best novel sentence from this paper
            if novel_sentences:
                contributions.append(novel_sentences[0][:200])
            elif abstract:
                # Fallback: use first substantial sentence
                first_sentence = sentences[0] if sentences else ""
                if len(first_sentence) > 30:
                    contributions.append(f"From {title.split(':')[0]}: {first_sentence[:150]}...")

        return contributions[:5]

    def _extract_technical_innovations(self, papers: List[Dict]) -> List[str]:
        """Extract technical innovations with performance metrics."""
        innovations = []

        for paper in papers:
            abstract = paper.get('abstract', '')

            # Look for quantitative improvements
            quantitative_findings = []
            for sentence in abstract.split('. '):
                if any(re.search(pattern, sentence) for pattern in self.technical_keywords['quantitative']):
                    clean_sentence = self._clean_sentence(sentence)
                    if len(clean_sentence) > 20:
                        quantitative_findings.append(clean_sentence)

            # Look for method descriptions
            method_sentences = []
            for sentence in abstract.split('. '):
                if any(keyword in sentence.lower() for keyword in
                      ['method', 'approach', 'algorithm', 'technique', 'architecture', 'framework']):
                    clean_sentence = self._clean_sentence(sentence)
                    if len(clean_sentence) > 20:
                        method_sentences.append(clean_sentence)

            # Prefer quantitative, fallback to methods
            best_innovation = (quantitative_findings + method_sentences + [""])[0]
            if best_innovation:
                innovations.append(best_innovation[:200])

        return innovations[:5]

    def _extract_business_implications(self, papers: List[Dict]) -> List[str]:
        """Extract business and commercial implications."""
        implications = []

        for paper in papers:
            abstract = paper.get('abstract', '')
            title = paper.get('title', '')

            # Look for business-relevant sentences
            business_sentences = []
            for sentence in abstract.split('. '):
                if any(keyword in sentence.lower() for keyword in self.business_keywords):
                    clean_sentence = self._clean_sentence(sentence)
                    if len(clean_sentence) > 20:
                        business_sentences.append(clean_sentence)

            # Look for application/impact sentences
            impact_sentences = []
            for sentence in abstract.split('. '):
                if any(term in sentence.lower() for term in
                      ['enables', 'allows', 'provides', 'offers', 'delivers', 'impact', 'benefit']):
                    clean_sentence = self._clean_sentence(sentence)
                    if len(clean_sentence) > 20:
                        impact_sentences.append(clean_sentence)

            best_implication = (business_sentences + impact_sentences + [""])[0]
            if best_implication:
                implications.append(best_implication[:200])
            elif title:
                # Generate business implication from title
                if 'embedding' in title.lower():
                    implications.append("Improved text embeddings enable better search and recommendation systems")
                elif 'productivity' in title.lower():
                    implications.append("Developer productivity insights inform AI tool adoption strategies")
                elif 'inference' in title.lower():
                    implications.append("Better statistical methods reduce data collection costs")

        return implications[:5]

    def _clean_sentence(self, sentence: str) -> str:
        """Clean and format a sentence for display."""
        # Remove extra whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        # Capitalize first letter
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]

        # Ensure it ends with proper punctuation
        if sentence and not sentence.endswith(('.', '!', '?')):
            sentence += '.'

        return sentence


def main():
    """Test the insight extractor."""
    # Load a sample report
    import json
    from pathlib import Path

    reports_dir = Path("reports/minimal_pipeline")
    if reports_dir.exists():
        json_files = list(reports_dir.glob("*.json"))
        if json_files:
            latest_report = max(json_files, key=lambda x: x.stat().st_mtime)

            with open(latest_report, 'r') as f:
                data = json.load(f)

            extractor = InsightExtractor()
            enhanced = extractor.enhance_analysis(data['analysis'], data['papers'])

            print("🔍 Enhanced Analysis Results:")
            print("\n📊 Novel Contributions:")
            for contrib in enhanced['novel_contributions']:
                print(f"  • {contrib}")

            print("\n⚡ Technical Innovations:")
            for innovation in enhanced['technical_innovations']:
                print(f"  • {innovation}")

            print("\n💼 Business Implications:")
            for implication in enhanced['business_implications']:
                print(f"  • {implication}")


if __name__ == "__main__":
    main()