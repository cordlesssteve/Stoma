"""
LLM-powered research analysis for intelligent content understanding.

This module provides cloud LLM integration for sophisticated research analysis,
serving as Phase 2A implementation toward the full hybrid LLM architecture.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os
import aiohttp

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from .nlp_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class LLMAnalysisResult:
    """Enhanced analysis result with LLM-generated insights."""
    
    document_id: str
    novel_contributions: List[str]
    research_significance: Dict[str, Any]
    methodology_assessment: Dict[str, Any]
    business_implications: List[str]
    technical_innovations: List[str]
    research_gaps_identified: List[str]
    related_work_connections: List[str]
    impact_prediction: Dict[str, Any]
    concept_keywords: List[Tuple[str, float]]
    research_quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMAnalyzer:
    """Cloud LLM integration for intelligent research analysis."""
    
    def __init__(self, 
                 provider: str = "openai",
                 model: str = "gpt-4",
                 max_tokens: int = 2000,
                 temperature: float = 0.1,
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Initialize LLM analyzer.
        
        Args:
            provider: LLM provider ("openai", "anthropic", "ollama")
            model: Model name to use
            max_tokens: Maximum tokens per response
            temperature: Response creativity (0.0-1.0)
            ollama_base_url: Base URL for Ollama API (for local models)
        """
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ollama_base_url = ollama_base_url
        
        # Initialize client
        self.client = None
        self._initialize_client()
        
        # Usage tracking
        self.usage_stats = {
            "total_requests": 0,
            "total_tokens": 0,
            "failed_requests": 0
        }
    
    def _initialize_client(self):
        """Initialize the appropriate LLM client."""
        if self.provider == "openai" and OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API key not found in environment")
                return
            self.client = openai.OpenAI(api_key=api_key)
            
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                logger.warning("Anthropic API key not found in environment")
                return
            self.client = anthropic.Anthropic(api_key=api_key)
            
        elif self.provider == "ollama":
            # For Ollama, we use HTTP client (aiohttp), no API key needed
            self.client = "ollama_http"  # Marker that we're using HTTP client
            logger.info(f"Initialized Ollama client with base URL: {self.ollama_base_url}")
            
        else:
            logger.warning(f"Provider {self.provider} not available or not supported")
    
    async def analyze_research_paper(self, 
                                   text: str, 
                                   title: str = "",
                                   document_id: str = "") -> LLMAnalysisResult:
        """
        Perform comprehensive LLM analysis of a research paper.
        
        Args:
            text: Full paper content
            title: Paper title
            document_id: Document identifier
            
        Returns:
            LLMAnalysisResult with intelligent insights
        """
        if not self.client:
            raise ValueError("LLM client not initialized. Check API keys and dependencies.")
        
        # For Ollama, check if model is available
        if self.provider == "ollama":
            model_available = await self._check_ollama_model_availability()
            if not model_available:
                raise ValueError(f"Ollama model '{self.model}' not available. Run: ollama pull {self.model}")
        
        # Prepare analysis prompts
        analysis_tasks = [
            self._analyze_novel_contributions(text, title),
            self._assess_research_significance(text, title),
            self._evaluate_methodology(text, title),
            self._identify_business_implications(text, title),
            self._extract_technical_innovations(text, title),
            self._predict_impact(text, title)
        ]
        
        # Execute analysis tasks
        try:
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Parse results
            novel_contributions = results[0] if not isinstance(results[0], Exception) else []
            research_significance = results[1] if not isinstance(results[1], Exception) else {}
            methodology_assessment = results[2] if not isinstance(results[2], Exception) else {}
            business_implications = results[3] if not isinstance(results[3], Exception) else []
            technical_innovations = results[4] if not isinstance(results[4], Exception) else []
            impact_prediction = results[5] if not isinstance(results[5], Exception) else {}
            
            # Generate concept keywords using LLM
            concept_keywords = await self._extract_concept_keywords(text, title)
            
            # Calculate overall quality score
            quality_score = self._calculate_research_quality_score(
                novel_contributions, research_significance, methodology_assessment
            )
            
            return LLMAnalysisResult(
                document_id=document_id,
                novel_contributions=novel_contributions,
                research_significance=research_significance,
                methodology_assessment=methodology_assessment,
                business_implications=business_implications,
                technical_innovations=technical_innovations,
                research_gaps_identified=[],  # Will implement in follow-up
                related_work_connections=[],  # Will implement in follow-up
                impact_prediction=impact_prediction,
                concept_keywords=concept_keywords,
                research_quality_score=quality_score,
                metadata={
                    "analyzed_at": datetime.now().isoformat(),
                    "llm_provider": self.provider,
                    "llm_model": self.model,
                    "analysis_version": "2.0_cloud_llm"
                }
            )
            
        except Exception as e:
            logger.error(f"LLM analysis failed for {document_id}: {e}")
            self.usage_stats["failed_requests"] += 1
            raise
    
    async def _analyze_novel_contributions(self, text: str, title: str) -> List[str]:
        """Identify novel contributions using LLM."""
        prompt = f"""
        Analyze this research paper and identify its novel contributions. Focus on what's genuinely new and innovative.

        Title: {title}

        Paper content: {text[:8000]}...

        Please identify 3-5 specific novel contributions this paper makes to the field. For each contribution:
        1. State what is novel/new
        2. Explain why it's significant
        3. How it differs from existing work

        Format as a JSON list of strings, each being a concise contribution description.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_list_response(response)
    
    async def _assess_research_significance(self, text: str, title: str) -> Dict[str, Any]:
        """Assess research significance and impact."""
        prompt = f"""
        Evaluate the research significance of this paper across multiple dimensions.

        Title: {title}
        Content: {text[:8000]}...

        Assess and score (1-10) the following aspects:
        1. Novelty: How new/original is the approach?
        2. Technical_rigor: How sound is the methodology?
        3. Practical_impact: How useful are the results?
        4. Theoretical_contribution: How much does it advance theory?
        5. Reproducibility: How well can results be reproduced?

        Format as JSON: {{"novelty": 8, "technical_rigor": 7, "practical_impact": 6, "theoretical_contribution": 9, "reproducibility": 5, "overall_assessment": "brief summary"}}
        """
        
        response = await self._call_llm(prompt)
        return self._parse_json_response(response)
    
    async def _evaluate_methodology(self, text: str, title: str) -> Dict[str, Any]:
        """Evaluate research methodology quality."""
        prompt = f"""
        Analyze the methodology of this research paper.

        Title: {title}
        Content: {text[:8000]}...

        Evaluate:
        1. Research design appropriateness
        2. Data collection methods
        3. Analysis techniques used
        4. Experimental setup quality
        5. Statistical validity
        6. Potential limitations

        Format as JSON: {{"design_quality": "assessment", "data_methods": "description", "analysis_techniques": ["list"], "experimental_setup": "evaluation", "statistical_validity": "assessment", "limitations": ["list"]}}
        """
        
        response = await self._call_llm(prompt)
        return self._parse_json_response(response)
    
    async def _identify_business_implications(self, text: str, title: str) -> List[str]:
        """Identify business and commercial implications."""
        prompt = f"""
        Analyze this research paper for business and commercial implications.

        Title: {title}
        Content: {text[:8000]}...

        Identify 3-5 potential business implications:
        1. Commercial applications
        2. Industry impact
        3. Market opportunities
        4. Competitive advantages
        5. Implementation challenges

        Format as JSON list of specific, actionable business implications.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_list_response(response)
    
    async def _extract_technical_innovations(self, text: str, title: str) -> List[str]:
        """Extract specific technical innovations."""
        prompt = f"""
        Identify specific technical innovations in this research paper.

        Title: {title}
        Content: {text[:8000]}...

        Extract:
        1. New algorithms or methods
        2. Novel architectures or designs
        3. Innovative applications of existing techniques
        4. Technical breakthroughs or insights
        5. New tools or frameworks introduced

        Format as JSON list of specific technical innovations with brief descriptions.
        """
        
        response = await self._call_llm(prompt)
        return self._parse_list_response(response)
    
    async def _predict_impact(self, text: str, title: str) -> Dict[str, Any]:
        """Predict research impact and influence."""
        prompt = f"""
        Predict the potential impact of this research.

        Title: {title}
        Content: {text[:8000]}...

        Assess:
        1. Short-term impact (1-2 years)
        2. Long-term impact (5+ years)
        3. Citation potential (high/medium/low)
        4. Industry adoption likelihood
        5. Research field influence

        Format as JSON: {{"short_term": "prediction", "long_term": "prediction", "citation_potential": "high/medium/low", "industry_adoption": "likelihood", "field_influence": "assessment", "impact_score": 7}}
        """
        
        response = await self._call_llm(prompt)
        return self._parse_json_response(response)
    
    async def _extract_concept_keywords(self, text: str, title: str) -> List[Tuple[str, float]]:
        """Extract conceptual keywords with semantic understanding."""
        prompt = f"""
        Extract the most important conceptual keywords from this research paper. Focus on:
        1. Key technical concepts
        2. Research methods and approaches
        3. Domain-specific terminology
        4. Innovative techniques mentioned
        5. Theoretical frameworks

        Title: {title}
        Content: {text[:8000]}...

        Extract 10-15 conceptual keywords with relevance scores (0.0-1.0).
        Format as JSON: {{"keywords": [["concept1", 0.9], ["concept2", 0.8], ...]}}
        """
        
        response = await self._call_llm(prompt)
        try:
            parsed = self._parse_json_response(response)
            return parsed.get("keywords", [])
        except:
            return []
    
    async def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM provider."""
        self.usage_stats["total_requests"] += 1
        
        try:
            if self.provider == "openai":
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                self.usage_stats["total_tokens"] += response.usage.total_tokens
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = await asyncio.to_thread(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                self.usage_stats["total_tokens"] += response.usage.input_tokens + response.usage.output_tokens
                return response.content[0].text
                
            elif self.provider == "ollama":
                return await self._call_ollama(prompt)
                
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            self.usage_stats["failed_requests"] += 1
            raise
    
    async def _call_ollama(self, prompt: str) -> str:
        """Make API call to Ollama local LLM."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                
                async with session.post(
                    f"{self.ollama_base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout for large models
                ) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama API returned status {response.status}: {await response.text()}")
                    
                    result = await response.json()
                    
                    # Track token usage (Ollama provides eval_count for output tokens)
                    output_tokens = result.get('eval_count', 0)
                    prompt_tokens = result.get('prompt_eval_count', 0)
                    self.usage_stats["total_tokens"] += output_tokens + prompt_tokens
                    
                    return result.get('response', '')
                    
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            self.usage_stats["failed_requests"] += 1
            raise
    
    async def _check_ollama_model_availability(self) -> bool:
        """Check if the specified model is available in Ollama."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.ollama_base_url}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        available_models = [model['name'] for model in result.get('models', [])]
                        
                        # Check exact match or partial match (e.g., llama3.1:70b matches llama3.1)
                        model_available = any(
                            self.model == model or 
                            self.model.split(':')[0] == model.split(':')[0]
                            for model in available_models
                        )
                        
                        if not model_available:
                            logger.warning(f"Model '{self.model}' not found in Ollama. Available models: {available_models}")
                            logger.info(f"To pull the model, run: ollama pull {self.model}")
                        
                        return model_available
                    else:
                        logger.error(f"Failed to connect to Ollama API at {self.ollama_base_url}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to check Ollama model availability: {e}")
            return False
    
    def _parse_list_response(self, response: str) -> List[str]:
        """Parse LLM response as list."""
        try:
            # Remove markdown code fences if present
            clean_response = response
            if "```json" in response:
                clean_response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                clean_response = response.split("```")[1].split("```")[0].strip()
            
            # Try to extract JSON from response
            if "[" in clean_response and "]" in clean_response:
                start = clean_response.find("[")
                end = clean_response.rfind("]") + 1
                json_str = clean_response[start:end]
                return json.loads(json_str)
            else:
                # Fallback: split by newlines and clean
                lines = [line.strip() for line in clean_response.split("\n") if line.strip()]
                return [line.lstrip("- ").lstrip("* ").lstrip("1234567890. ") for line in lines if line]
        except Exception as e:
            logger.warning(f"Failed to parse list response: {e}")
            # Return empty list on parsing failure
            return []
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response as JSON."""
        try:
            # Remove markdown code fences if present
            clean_response = response
            if "```json" in response:
                clean_response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                clean_response = response.split("```")[1].split("```")[0].strip()
            
            # Try to extract JSON from response
            if "{" in clean_response and "}" in clean_response:
                start = clean_response.find("{")
                end = clean_response.rfind("}") + 1
                json_str = clean_response[start:end]
                return json.loads(json_str)
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return {}
    
    def _calculate_research_quality_score(self, 
                                        contributions: List[str],
                                        significance: Dict[str, Any],
                                        methodology: Dict[str, Any]) -> float:
        """Calculate overall research quality score."""
        try:
            # Base score from significance ratings
            base_score = 0.0
            if significance:
                scores = [
                    significance.get("novelty", 5),
                    significance.get("technical_rigor", 5),
                    significance.get("practical_impact", 5),
                    significance.get("theoretical_contribution", 5)
                ]
                base_score = sum(scores) / len(scores) / 10.0  # Normalize to 0-1
            
            # Bonus for number of contributions
            contribution_bonus = min(len(contributions) * 0.1, 0.3)
            
            # Methodology bonus (simplified)
            methodology_bonus = 0.1 if methodology else 0.0
            
            final_score = min(base_score + contribution_bonus + methodology_bonus, 1.0)
            return round(final_score, 2)
            
        except Exception:
            return 0.5  # Default middle score
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            **self.usage_stats,
            "success_rate": (self.usage_stats["total_requests"] - self.usage_stats["failed_requests"]) / max(self.usage_stats["total_requests"], 1)
        }


class LLMAnalysisService:
    """Service for managing LLM analysis across multiple documents."""
    
    def __init__(self, analyzer: LLMAnalyzer):
        self.analyzer = analyzer
        self.analysis_cache = {}
    
    async def analyze_multiple_papers(self, 
                                    papers: List[Dict[str, Any]],
                                    enable_cross_analysis: bool = True) -> List[LLMAnalysisResult]:
        """
        Analyze multiple papers with optional cross-document analysis.
        
        Args:
            papers: List of paper dicts with 'content', 'title', 'document_id'
            enable_cross_analysis: Whether to perform cross-document analysis
            
        Returns:
            List of LLMAnalysisResult objects
        """
        results = []
        
        # Analyze each paper individually
        for paper in papers:
            try:
                result = await self.analyzer.analyze_research_paper(
                    text=paper["content"],
                    title=paper.get("title", ""),
                    document_id=paper.get("document_id", "")
                )
                results.append(result)
                
                # Cache for cross-analysis
                self.analysis_cache[paper.get("document_id", "")] = result
                
            except Exception as e:
                logger.error(f"Failed to analyze paper {paper.get('document_id', '')}: {e}")
        
        # Perform cross-document analysis if enabled
        if enable_cross_analysis and len(results) > 1:
            await self._perform_cross_analysis(results)
        
        return results
    
    async def _perform_cross_analysis(self, results: List[LLMAnalysisResult]):
        """Perform cross-document analysis to identify trends and connections."""
        # This will be expanded in the next phase
        logger.info(f"Cross-analysis capability planned for {len(results)} papers")
        pass
    
    def generate_research_intelligence_summary(self, 
                                             results: List[LLMAnalysisResult]) -> Dict[str, Any]:
        """Generate high-level intelligence summary from multiple analyses."""
        if not results:
            return {}
        
        # Aggregate insights
        all_contributions = []
        all_innovations = []
        all_implications = []
        quality_scores = []
        
        for result in results:
            all_contributions.extend(result.novel_contributions)
            all_innovations.extend(result.technical_innovations)
            all_implications.extend(result.business_implications)
            quality_scores.append(result.research_quality_score)
        
        return {
            "total_papers_analyzed": len(results),
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "total_novel_contributions": len(all_contributions),
            "top_innovations": all_innovations[:10],
            "key_business_implications": all_implications[:10],
            "analysis_timestamp": datetime.now().isoformat()
        }