"""Traditional NLP analysis pipeline for academic papers and documents."""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter
import math

try:
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    STOP_WORDS = set()

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Container for NLP analysis results."""
    
    document_id: str
    summary: str
    keywords: List[Tuple[str, float]]
    entities: Dict[str, List[str]]
    sentiment: Dict[str, float]
    topics: List[str]
    readability_score: float
    word_count: int
    sentence_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class NLPAnalyzer:
    """Traditional NLP analysis pipeline for documents."""
    
    def __init__(self, 
                 language: str = "en",
                 spacy_model: str = "en_core_web_sm",
                 max_summary_sentences: int = 5,
                 max_keywords: int = 10):
        """
        Initialize NLP analyzer with traditional methods.
        
        Args:
            language: Language code for analysis
            spacy_model: SpaCy model to use
            max_summary_sentences: Maximum sentences in summary
            max_keywords: Maximum keywords to extract
        """
        self.language = language
        self.max_summary_sentences = max_summary_sentences
        self.max_keywords = max_keywords
        
        # Initialize NLP components
        self._initialize_nltk()
        self._initialize_spacy(spacy_model)
        
        # Initialize tools
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
    def _initialize_nltk(self):
        """Download required NLTK data."""
        required_data = [
            ('punkt', 'tokenizers/punkt'),
            ('punkt_tab', 'tokenizers/punkt_tab'),
            ('stopwords', 'corpora/stopwords'),
            ('wordnet', 'corpora/wordnet'), 
            ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
            ('omw-1.4', 'corpora/omw-1.4')
        ]
        
        for data_name, data_path in required_data:
            try:
                nltk.data.find(data_path)
            except LookupError:
                try:
                    logger.info(f"Downloading NLTK data: {data_name}")
                    nltk.download(data_name, quiet=True)
                except Exception as e:
                    logger.warning(f"Failed to download {data_name}: {e}")
    
    def _initialize_spacy(self, model_name: str):
        """Load SpaCy model."""
        if not SPACY_AVAILABLE:
            logger.warning("SpaCy not available, using fallback methods")
            self.nlp = None
            return
            
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"SpaCy model {model_name} not found, using fallback methods")
            self.nlp = None
    
    def analyze(self, text: str, document_id: str = "") -> AnalysisResult:
        """
        Perform comprehensive NLP analysis on text.
        
        Args:
            text: Document text to analyze
            document_id: Optional document identifier
            
        Returns:
            AnalysisResult with all analysis outputs
        """
        # Basic statistics
        word_count = len(word_tokenize(text))
        sentences = sent_tokenize(text)
        sentence_count = len(sentences)
        
        # Perform analyses
        summary = self.extractive_summarization(text)
        keywords = self.extract_keywords(text)
        entities = self.extract_entities(text)
        sentiment = self.analyze_sentiment(text)
        topics = self.extract_topics(text)
        readability = self.calculate_readability(text)
        
        return AnalysisResult(
            document_id=document_id,
            summary=summary,
            keywords=keywords,
            entities=entities,
            sentiment=sentiment,
            topics=topics,
            readability_score=readability,
            word_count=word_count,
            sentence_count=sentence_count
        )
    
    def extractive_summarization(self, text: str) -> str:
        """
        Extract key sentences using TF-IDF scoring.
        
        Args:
            text: Document text
            
        Returns:
            Summary text
        """
        sentences = sent_tokenize(text)
        
        if len(sentences) <= self.max_summary_sentences:
            return text
        
        # Calculate sentence scores using TF-IDF
        word_frequencies = self._calculate_word_frequencies(text)
        sentence_scores = {}
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            
            for word in words:
                if word in word_frequencies:
                    score += word_frequencies[word]
                    word_count += 1
            
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Select top sentences
        ranked_sentences = sorted(
            sentence_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_summary_sentences]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if any(sentence == s[0] for s in ranked_sentences):
                summary_sentences.append(sentence)
        
        return ' '.join(summary_sentences)
    
    def _calculate_word_frequencies(self, text: str) -> Dict[str, float]:
        """Calculate TF-IDF word frequencies."""
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter and count words
        word_counts = Counter(
            word for word in words 
            if word.isalpha() and word not in stop_words
        )
        
        # Calculate TF-IDF scores
        total_words = sum(word_counts.values())
        word_frequencies = {}
        
        for word, count in word_counts.items():
            tf = count / total_words
            # Simple IDF approximation
            idf = math.log(1 + 1 / (1 + count))
            word_frequencies[word] = tf * idf
        
        return word_frequencies
    
    def extract_keywords(self, text: str) -> List[Tuple[str, float]]:
        """
        Extract keywords using multiple methods.
        
        Args:
            text: Document text
            
        Returns:
            List of (keyword, score) tuples
        """
        # Method 1: TF-IDF based keywords
        tfidf_keywords = self._extract_tfidf_keywords(text)
        
        # Method 2: TextRank (using spaCy)
        textrank_keywords = self._extract_textrank_keywords(text)
        
        # Method 3: Noun phrases
        noun_phrases = self._extract_noun_phrases(text)
        
        # Combine and rank
        all_keywords = {}
        for keyword, score in tfidf_keywords:
            all_keywords[keyword] = score
        
        for keyword, score in textrank_keywords:
            if keyword in all_keywords:
                all_keywords[keyword] = (all_keywords[keyword] + score) / 2
            else:
                all_keywords[keyword] = score * 0.8
        
        for phrase in noun_phrases[:20]:
            if phrase not in all_keywords:
                all_keywords[phrase] = 0.5
        
        # Sort and return top keywords
        sorted_keywords = sorted(
            all_keywords.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:self.max_keywords]
        
        return sorted_keywords
    
    def _extract_tfidf_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using TF-IDF."""
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            keywords = [
                (feature_names[i], scores[i]) 
                for i in scores.argsort()[-self.max_keywords:][::-1]
                if scores[i] > 0
            ]
            
            return keywords
        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            return []
    
    def _extract_textrank_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Extract keywords using TextRank algorithm."""
        if self.nlp is None:
            # Fallback to simple word frequency
            return self._extract_frequency_keywords(text)
            
        doc = self.nlp(text)
        
        # Filter tokens
        candidates = []
        for token in doc:
            if (token.is_alpha and 
                not token.is_stop and 
                token.pos_ in ['NOUN', 'PROPN', 'ADJ']):
                candidates.append(token.lemma_.lower())
        
        # Simple frequency-based scoring as TextRank approximation
        word_freq = Counter(candidates)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        keywords = [
            (word, freq/max_freq) 
            for word, freq in word_freq.most_common(self.max_keywords)
        ]
        
        return keywords
    
    def _extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases from text."""
        if self.nlp is None:
            # Fallback to simple ngram extraction
            return self._extract_ngrams(text)
            
        doc = self.nlp(text)
        
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Clean and filter noun phrases
            phrase = chunk.text.lower().strip()
            if len(phrase.split()) <= 4 and len(phrase) > 3:
                noun_phrases.append(phrase)
        
        # Return unique phrases
        return list(dict.fromkeys(noun_phrases))
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities using SpaCy NER.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {}
        
        if self.nlp is not None:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = ent.label_
                entity_text = ent.text
                
                if entity_type not in entities:
                    entities[entity_type] = []
                
                if entity_text not in entities[entity_type]:
                    entities[entity_type].append(entity_text)
        
        # Always extract custom academic entities (pattern-based)
        academic_entities = self._extract_academic_entities(text)
        entities.update(academic_entities)
        
        return entities
    
    def _extract_academic_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract academic-specific entities."""
        entities = {
            'ALGORITHM': [],
            'DATASET': [],
            'METRIC': [],
            'METHOD': []
        }
        
        # Pattern-based extraction for academic entities
        patterns = {
            'ALGORITHM': r'\b(?:algorithm|method|approach|technique|model)\s+(?:called|named|termed)?\s*([A-Z][A-Za-z0-9\-]+)',
            'DATASET': r'\b(?:dataset|corpus|benchmark)\s+(?:called|named)?\s*([A-Z][A-Za-z0-9\-]+)',
            'METRIC': r'\b([A-Z][A-Za-z0-9\-]+)\s+(?:score|metric|measure|accuracy|precision|recall|f1)',
            'METHOD': r'\b(?:propose|introduce|present)\s+(?:a|an|the)?\s*(?:new|novel)?\s*([A-Za-z0-9\-]+)'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities[entity_type] = list(set(matches))[:5]
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with sentiment scores
        """
        blob = TextBlob(text)
        
        # Overall sentiment
        sentiment = {
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
            'sentiment_label': self._get_sentiment_label(blob.sentiment.polarity)
        }
        
        # Sentence-level sentiment distribution
        sentences = sent_tokenize(text)
        sentence_sentiments = []
        
        for sentence in sentences[:50]:  # Analyze first 50 sentences
            sent_blob = TextBlob(sentence)
            sentence_sentiments.append(sent_blob.sentiment.polarity)
        
        if sentence_sentiments:
            sentiment['avg_sentence_polarity'] = np.mean(sentence_sentiments)
            sentiment['std_sentence_polarity'] = np.std(sentence_sentiments)
            sentiment['positive_sentences'] = sum(1 for s in sentence_sentiments if s > 0.1)
            sentiment['negative_sentences'] = sum(1 for s in sentence_sentiments if s < -0.1)
            sentiment['neutral_sentences'] = sum(1 for s in sentence_sentiments if -0.1 <= s <= 0.1)
        
        return sentiment
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert polarity score to label."""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_topics(self, text: str) -> List[str]:
        """
        Extract main topics using simple topic modeling.
        
        Args:
            text: Document text
            
        Returns:
            List of topic labels
        """
        if self.nlp is not None:
            # Use SpaCy for noun phrase extraction
            doc = self.nlp(text)
            
            # Extract significant noun phrases
            noun_phrases = [
                chunk.text.lower() 
                for chunk in doc.noun_chunks 
                if len(chunk.text.split()) >= 2
            ]
        else:
            # Fallback to simple ngram extraction
            noun_phrases = self._extract_ngrams(text, min_length=2)
        
        # Count and rank phrases
        phrase_counts = Counter(noun_phrases)
        
        # Get top phrases as topics
        topics = [
            phrase for phrase, _ in phrase_counts.most_common(5)
        ]
        
        return topics
    
    def calculate_readability(self, text: str) -> float:
        """
        Calculate Flesch Reading Ease score.
        
        Args:
            text: Document text
            
        Returns:
            Readability score (0-100, higher is easier)
        """
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        if not sentences or not words:
            return 0.0
        
        # Count syllables
        syllable_count = sum(self._count_syllables(word) for word in words if word.isalpha())
        
        # Flesch Reading Ease formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = syllable_count / len(words)
        
        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
        
        # Clamp between 0 and 100
        return max(0, min(100, flesch_score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Ensure at least one syllable
        if syllable_count == 0:
            syllable_count = 1
        
        return syllable_count
    
    def _extract_frequency_keywords(self, text: str) -> List[Tuple[str, float]]:
        """Fallback keyword extraction using word frequency."""
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english') if nltk.data.find('corpora/stopwords') else [])
        
        # Filter words
        filtered_words = [
            word for word in words 
            if word.isalpha() and word not in stop_words and len(word) > 3
        ]
        
        # Count frequencies
        word_freq = Counter(filtered_words)
        max_freq = max(word_freq.values()) if word_freq else 1
        
        # Return normalized scores
        keywords = [
            (word, freq/max_freq) 
            for word, freq in word_freq.most_common(self.max_keywords)
        ]
        
        return keywords
    
    def _extract_ngrams(self, text: str, min_length: int = 1) -> List[str]:
        """Extract n-grams from text as fallback for noun phrases."""
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english') if nltk.data.find('corpora/stopwords') else [])
        
        # Filter words
        filtered_words = [
            word for word in words 
            if word.isalpha() and word not in stop_words
        ]
        
        # Generate bigrams and trigrams
        ngrams = []
        
        for i in range(len(filtered_words) - min_length + 1):
            for n in range(min_length, min(4, len(filtered_words) - i + 1)):
                ngram = ' '.join(filtered_words[i:i+n])
                if len(ngram) > 3:
                    ngrams.append(ngram)
        
        return ngrams[:20]  # Return top 20