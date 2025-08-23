#!/usr/bin/env python3
"""
Evaluation Metrics Implementation
Based on Mathematical Formulas Evaluation Metrics.md
Includes BLEU, ROUGE, METEOR, BERTScore, and GEMBA implementations
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
import re
from collections import Counter
import math

class TranslationEvaluationMetrics:
    """
    Implementation of translation quality evaluation metrics
    Based on the mathematical formulas from the research paper
    """
    
    def __init__(self):
        pass
    
    def bleu_score(self, candidate, reference, max_n=4, weights=None):
        """
        BLEU Score Implementation
        Formula: BLEU = BP × exp(∑(w_n × log p_n))
        """
        if weights is None:
            weights = [0.25] * max_n
        
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        
        # Calculate brevity penalty
        bp = self.brevity_penalty(candidate_tokens, reference_tokens)
        
        # Calculate n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            precision = self.modified_precision(candidate_tokens, reference_tokens, n)
            if precision > 0:
                precisions.append(precision)
            else:
                # Handle zero precision case
                return 0.0
        
        if len(precisions) == 0:
            return 0.0
        
        # Calculate BLEU score
        log_precisions = [w * math.log(p) for w, p in zip(weights[:len(precisions)], precisions)]
        bleu = bp * math.exp(sum(log_precisions))
        
        return bleu
    
    def brevity_penalty(self, candidate, reference):
        """Brevity Penalty: BP = min(1, exp(1 - r/c))"""
        c = len(candidate)  # candidate length
        r = len(reference)  # reference length
        
        if c == 0:
            return 0
        if c > r:
            return 1.0
        else:
            return math.exp(1 - r/c)
    
    def modified_precision(self, candidate, reference, n):
        """Modified n-gram precision for BLEU"""
        candidate_ngrams = self.get_ngrams(candidate, n)
        reference_ngrams = self.get_ngrams(reference, n)
        
        if len(candidate_ngrams) == 0:
            return 0.0
        
        # Count clipped n-grams
        clipped_count = 0
        for ngram in candidate_ngrams:
            clipped_count += min(candidate_ngrams[ngram], reference_ngrams.get(ngram, 0))
        
        return clipped_count / sum(candidate_ngrams.values())
    
    def get_ngrams(self, tokens, n):
        """Extract n-grams from token list"""
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def rouge_l_score(self, candidate, reference, beta=1.0):
        """
        ROUGE-L Score Implementation  
        Formula: ROUGE-L = F_lcs = ((1 + β²) × R_lcs × P_lcs) / (R_lcs + β² × P_lcs)
        """
        candidate_tokens = candidate.split()
        reference_tokens = reference.split()
        
        # Calculate LCS length
        lcs_length = self.lcs_length(candidate_tokens, reference_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = lcs_length / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
        recall = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0
        
        # Calculate F-score
        if precision + recall == 0:
            return 0.0
        
        f_score = ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)
        return f_score
    
    def lcs_length(self, seq1, seq2):
        """Longest Common Subsequence length using dynamic programming"""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def meteor_score(self, candidate, reference, alpha=0.9, beta=3, gamma=0.5):
        """
        METEOR Score Implementation (simplified version)
        Incorporates precision, recall, and word order penalties
        """
        candidate_tokens = candidate.lower().split()
        reference_tokens = reference.lower().split()
        
        # Calculate exact matches
        matches = self.count_matches(candidate_tokens, reference_tokens)
        
        if matches == 0:
            return 0.0
        
        # Calculate precision and recall
        precision = matches / len(candidate_tokens) if len(candidate_tokens) > 0 else 0
        recall = matches / len(reference_tokens) if len(reference_tokens) > 0 else 0
        
        if precision == 0 and recall == 0:
            return 0.0
        
        # Calculate harmonic mean (F-score)
        f_score = (10 * precision * recall) / (9 * precision + recall) if (precision + recall) > 0 else 0
        
        # Penalty for word order differences (simplified)
        penalty = gamma * (matches / len(candidate_tokens))**beta if len(candidate_tokens) > 0 else 0
        
        meteor = f_score * (1 - penalty)
        return max(0, meteor)
    
    def count_matches(self, candidate, reference):
        """Count exact word matches between candidate and reference"""
        ref_counts = Counter(reference)
        matches = 0
        
        for word in candidate:
            if word in ref_counts and ref_counts[word] > 0:
                matches += 1
                ref_counts[word] -= 1
        
        return matches
    
    def chrf_score(self, candidate, reference, n=6, beta=2):
        """
        chrF Score Implementation
        Character n-gram F-score for automatic MT evaluation
        """
        candidate_chars = list(candidate.replace(' ', ''))
        reference_chars = list(reference.replace(' ', ''))
        
        f_scores = []
        
        for i in range(1, n + 1):
            # Get character n-grams
            cand_ngrams = self.get_char_ngrams(candidate_chars, i)
            ref_ngrams = self.get_char_ngrams(reference_chars, i)
            
            if len(cand_ngrams) == 0 and len(ref_ngrams) == 0:
                f_scores.append(1.0)
                continue
            elif len(cand_ngrams) == 0 or len(ref_ngrams) == 0:
                f_scores.append(0.0)
                continue
            
            # Calculate precision and recall
            matches = sum(min(cand_ngrams.get(ng, 0), ref_ngrams.get(ng, 0)) for ng in set(cand_ngrams) | set(ref_ngrams))
            precision = matches / sum(cand_ngrams.values())
            recall = matches / sum(ref_ngrams.values())
            
            # Calculate F-score
            if precision + recall == 0:
                f_scores.append(0.0)
            else:
                f_score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
                f_scores.append(f_score)
        
        # Average F-scores
        return sum(f_scores) / len(f_scores) if f_scores else 0.0
    
    def get_char_ngrams(self, chars, n):
        """Extract character n-grams"""
        ngrams = Counter()
        for i in range(len(chars) - n + 1):
            ngram = ''.join(chars[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def gemba_score_simulation(self, candidate, reference):
        """
        Simulated GEMBA Score (GPT-based evaluation)
        Real implementation would require LLM API calls
        This provides a correlation-based approximation
        """
        # Simulate LLM evaluation based on other metrics
        bleu = self.bleu_score(candidate, reference)
        rouge = self.rouge_l_score(candidate, reference)
        chrf = self.chrf_score(candidate, reference)
        
        # Weighted combination to simulate GEMBA-style evaluation
        # Higher correlation with human judgments
        simulated_gemba = (0.3 * bleu + 0.4 * rouge + 0.3 * chrf)
        
        # Convert to 1-5 scale (typical GEMBA output)
        gemba_5_scale = 1 + (simulated_gemba * 4)
        
        return min(5.0, max(1.0, gemba_5_scale))
    
    def evaluate_translation_pair(self, candidate, reference):
        """
        Comprehensive evaluation using all metrics
        Returns dictionary with all scores
        """
        results = {
            'BLEU': self.bleu_score(candidate, reference),
            'ROUGE-L': self.rouge_l_score(candidate, reference),
            'METEOR': self.meteor_score(candidate, reference),
            'chrF': self.chrf_score(candidate, reference),
            'GEMBA': self.gemba_score_simulation(candidate, reference)
        }
        
        return results
    
    def batch_evaluate(self, candidates, references):
        """Evaluate multiple translation pairs"""
        if len(candidates) != len(references):
            raise ValueError("Number of candidates must match number of references")
        
        results = []
        for cand, ref in zip(candidates, references):
            score_dict = self.evaluate_translation_pair(cand, ref)
            results.append(score_dict)
        
        # Convert to DataFrame for easy analysis
        df = pd.DataFrame(results)
        return df

def demo_evaluation():
    """Demonstration of the evaluation metrics"""
    print("🔬 TRANSLATION EVALUATION METRICS DEMO")
    print("=" * 60)
    
    evaluator = TranslationEvaluationMetrics()
    
    # Swedish-English example translations
    examples = [
        {
            'reference': "The weather is beautiful today and perfect for a walk in the park.",
            'candidate_human': "The weather is lovely today and ideal for a stroll in the park.",
            'candidate_llm': "Weather is nice today and good for walk in park.",
            'candidate_poor': "The sunny is today very much walking."
        },
        {
            'reference': "Sweden has made significant progress in renewable energy development.",
            'candidate_human': "Sweden has achieved considerable advancement in renewable energy development.", 
            'candidate_llm': "Sweden has made major progress in developing renewable energy.",
            'candidate_poor': "Sweden progress renewable energy significant."
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"\n📝 Example {i+1}:")
        print(f"Reference: {example['reference']}")
        print(f"Human: {example['candidate_human']}")
        print(f"LLM: {example['candidate_llm']}")
        print(f"Poor: {example['candidate_poor']}")
        print()
        
        # Evaluate each candidate
        for candidate_type in ['candidate_human', 'candidate_llm', 'candidate_poor']:
            candidate = example[candidate_type]
            scores = evaluator.evaluate_translation_pair(candidate, example['reference'])
            
            print(f"{candidate_type.replace('candidate_', '').title()} scores:")
            for metric, score in scores.items():
                if metric == 'GEMBA':
                    print(f"   {metric}: {score:.2f}/5.0")
                else:
                    print(f"   {metric}: {score:.3f}")
            print()
    
    print("✅ Evaluation metrics demonstration completed!")
    print("These metrics align with the research paper methodology.")

if __name__ == "__main__":
    demo_evaluation()
