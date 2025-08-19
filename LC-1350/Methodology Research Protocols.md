# Appendix A: Methodology and Research Protocols

## A1. Human Evaluation Protocol

### A1.1 Translator Selection Criteria

**Primary Qualifications:**
- Certified Swedish-English translator (minimum 5 years professional experience)
- Native or near-native proficiency in both languages
- Experience with CAT tools and quality assessment
- No prior exposure to the specific test dataset

**Selection Process:**
1. Initial screening via professional translation associations
2. Portfolio review and reference checks
3. Qualification test using standardized materials
4. Agreement to participate with informed consent

**Final Pool Composition:**
- 15 certified translators selected
- Age range: 28-55 years
- Gender distribution: 60% female, 40% male
- Geographic distribution: 40% Sweden, 35% US/UK, 25% other English-speaking countries

### A1.2 Translation Task Protocol

**Dataset Preparation:**
- 1,000 sentence pairs per translation direction (Swedish→English, English→Swedish)
- Sentence length: 10-50 words (controlled for complexity)
- Domain distribution: 40% general, 20% news, 15% technical, 15% business, 10% cultural content
- Random selection from validated corpora (OPUS, Europarl, custom datasets)

**Translation Procedure:**
1. **Session Setup**: Controlled environment, standardized CAT tool (SDL Trados Studio)
2. **Time Allocation**: Maximum 4 hours per session, with breaks
3. **Reference Materials**: Standard dictionaries and glossaries permitted
4. **Quality Control**: No internet access, no collaboration between translators

**Data Collection:**
- Translation output for each sentence pair
- Time stamps for productivity analysis
- Keystroke logging for effort measurement
- Post-task questionnaire on difficulty assessment

### A1.3 Quality Assessment Framework

**Evaluation Dimensions:**
1. **Adequacy** (1-5 scale): Semantic accuracy and completeness
2. **Fluency** (1-5 scale): Grammatical correctness and naturalness
3. **Cultural Appropriateness** (1-3 scale): Context-sensitive adaptation
4. **Terminology** (1-3 scale): Specialized vocabulary accuracy

**Inter-Annotator Reliability:**
- Double-blind evaluation by 3 independent assessors
- Krippendorff's Alpha calculation for agreement measurement
- Consensus sessions for disagreements > 1 point on 5-point scale
- Final scores averaged across annotators with weighted confidence intervals

**Assessment Protocol:**
```
For each translated sentence:
1. Assess adequacy: Does the translation convey the same meaning?
2. Assess fluency: Is the translation natural and grammatically correct?
3. Assess terminology: Are technical terms correctly translated?
4. Assess cultural adaptation: Are cultural references appropriate?
5. Provide overall quality score (1-5)
6. Note specific issues for post-hoc analysis
```

## A2. LLM Evaluation Protocol

### A2.1 Model Selection and Configuration

**Primary Models Evaluated:**
1. **GPT-4** (OpenAI)
   - Model: gpt-4-0314 (consistent version)
   - Temperature: 0.1 (low variability)
   - Max tokens: 2048
   - API configuration: Standard parameters

2. **Claude-3** (Anthropic)
   - Model: claude-3-opus-20240229
   - Temperature: 0.0 (deterministic)
   - Max tokens: 2048
   - System prompt: Translation-optimized

3. **Gemini Pro** (Google)
   - Model: gemini-pro-001
   - Temperature: 0.1
   - Safety settings: Disabled for academic research
   - Configuration: Multilingual mode enabled

**Specialized Translation Models:**
4. **NLLB-200** (Meta)
   - Model: facebook/nllb-200-3.3B
   - Configuration: Swedish-English optimized
   - Beam search: width=5

5. **mT5-XXL** (Google)
   - Model: google/mt5-xxl
   - Fine-tuned on Swedish-English parallel corpus
   - Beam search: width=5, length penalty=1.0

### A2.2 Translation Generation Protocol

**Prompt Engineering:**
- **General Purpose LLMs**: Standardized prompt template
- **Task Specificity**: Clear instruction for Swedish↔English translation
- **Context Preservation**: Minimal additional context to avoid bias
- **Output Format**: Plain text translation only

**Standard Prompt Template:**
```
Translate the following text from [SOURCE_LANGUAGE] to [TARGET_LANGUAGE]. 
Provide only the translation without additional commentary.

Text to translate: [INPUT_TEXT]

Translation:
```

**Quality Control Measures:**
- Multiple runs per sentence to assess consistency (n=3)
- API rate limiting to avoid service degradation
- Error handling for API failures or incomplete responses
- Validation of output language and format

### A2.3 Automatic Evaluation Implementation

**BLEU Score Calculation:**
```python
from sacrebleu import BLEU

def calculate_bleu(candidates, references):
    """
    Calculate BLEU scores with standard parameters
    """
    bleu = BLEU(effective_order=True)
    score = bleu.corpus_score(candidates, [references])
    return score.score, score.counts, score.totals
```

**ROUGE-L Implementation:**
```python
from rouge import Rouge

def calculate_rouge_l(candidates, references):
    """
    Calculate ROUGE-L scores for translation evaluation
    """
    rouge = Rouge()
    scores = rouge.get_scores(candidates, references, avg=True)
    return scores['rouge-l']['f']
```

**BERTScore Configuration:**
```python
from bert_score import score

def calculate_bertscore(candidates, references, lang='en'):
    """
    Calculate BERTScore with specified language model
    """
    P, R, F1 = score(candidates, references, lang=lang, 
                     model_type='microsoft/deberta-xlarge-mnli',
                     num_layers=40)
    return P.mean().item(), R.mean().item(), F1.mean().item()
```

**GEMBA Implementation:**
```python
def gemba_evaluation(source, candidate, reference, model='gpt-4'):
    """
    Implement GEMBA scoring using LLM evaluation
    """
    prompt = f"""
    Evaluate the quality of this translation on a scale of 1-5:
    
    Source ({source_lang}): {source}
    Translation: {candidate}
    Reference: {reference}
    
    Consider accuracy, fluency, and naturalness.
    Score (1-5):
    """
    
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    return extract_score(response.choices[0].message.content)
```

## A3. Statistical Analysis Procedures

### A3.1 Descriptive Statistics

**Performance Metrics:**
- Mean, median, standard deviation for all evaluation metrics
- Confidence intervals (95%) using bootstrap methods
- Distribution analysis (normality testing via Shapiro-Wilk)
- Outlier detection using IQR method

**Comparative Analysis:**
```python
from scipy import stats
import numpy as np

def comparative_analysis(human_scores, llm_scores):
    """
    Perform statistical comparison between human and LLM performance
    """
    # Descriptive statistics
    human_stats = {
        'mean': np.mean(human_scores),
        'std': np.std(human_scores),
        'median': np.median(human_scores),
        'ci_95': stats.norm.interval(0.95, np.mean(human_scores), 
                                   stats.sem(human_scores))
    }
    
    llm_stats = {
        'mean': np.mean(llm_scores),
        'std': np.std(llm_scores),
        'median': np.median(llm_scores),
        'ci_95': stats.norm.interval(0.95, np.mean(llm_scores), 
                                   stats.sem(llm_scores))
    }
    
    # Statistical tests
    t_stat, p_value = stats.ttest_ind(human_scores, llm_scores)
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(human_scores, llm_scores)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(human_scores)**2 + np.std(llm_scores)**2) / 2)
    cohens_d = (np.mean(human_scores) - np.mean(llm_scores)) / pooled_std
    
    return human_stats, llm_stats, t_stat, p_value, cohens_d
```

### A3.2 Correlation Analysis

**Metric Validation:**
- Pearson correlation between automatic metrics and human judgments
- Spearman rank correlation for non-parametric relationships
- Kendall's tau for concordance assessment
- Partial correlation controlling for sentence length and complexity

**Inter-Annotator Reliability:**
```python
import krippendorff

def calculate_reliability(annotations):
    """
    Calculate inter-annotator reliability using Krippendorff's Alpha
    """
    # Convert annotations to appropriate format
    reliability_data = prepare_reliability_data(annotations)
    
    # Calculate Krippendorff's Alpha
    alpha = krippendorff.alpha(reliability_data, 
                              level_of_measurement='interval')
    
    return alpha

def fleiss_kappa(annotations):
    """
    Calculate Fleiss' Kappa for multiple raters
    """
    # Implementation of Fleiss' Kappa calculation
    n_items, n_raters = annotations.shape
    categories = np.unique(annotations)
    
    # Calculate agreement matrix
    agreement_matrix = calculate_agreement_matrix(annotations, categories)
    
    # Calculate kappa statistic
    kappa = fleiss_kappa_statistic(agreement_matrix)
    
    return kappa
```

### A3.3 Time Series Analysis

**Performance Trajectory Modeling:**
```python
from scipy.optimize import curve_fit
import numpy as np

def logistic_growth(t, Q_max, k, t_0):
    """Logistic growth model for quality improvement"""
    return Q_max / (1 + np.exp(-k * (t - t_0)))

def gompertz_growth(t, Q_max, k, t_0):
    """Gompertz growth model for quality improvement"""
    return Q_max * np.exp(-np.exp(-k * (t - t_0)))

def fit_growth_model(time_points, quality_scores, model='logistic'):
    """
    Fit growth models to historical performance data
    """
    if model == 'logistic':
        popt, pcov = curve_fit(logistic_growth, time_points, quality_scores,
                              p0=[0.95, 0.5, 4.0])
    elif model == 'gompertz':
        popt, pcov = curve_fit(gompertz_growth, time_points, quality_scores,
                              p0=[0.95, 0.5, 4.0])
    
    # Calculate goodness of fit
    y_pred = model_function(time_points, *popt)
    r_squared = 1 - np.sum((quality_scores - y_pred)**2) / np.sum((quality_scores - np.mean(quality_scores))**2)
    
    return popt, pcov, r_squared

def predict_timeline(model_params, target_quality=0.9):
    """
    Predict when target quality will be achieved
    """
    Q_max, k, t_0 = model_params
    
    if model == 'logistic':
        t_target = t_0 - (1/k) * np.log((Q_max/target_quality) - 1)
    elif model == 'gompertz':
        t_target = t_0 - (1/k) * np.log(-np.log(target_quality/Q_max))
    
    return t_target
```

## A4. Monte Carlo Simulation Implementation

### A4.1 Parameter Uncertainty Modeling

**Distribution Specifications:**
```python
import numpy as np
from scipy import stats

class ParameterDistributions:
    """
    Define probability distributions for uncertain parameters
    """
    
    def __init__(self):
        # Quality improvement rate (annual)
        self.quality_rate = stats.norm(loc=0.15, scale=0.05)
        
        # Economic adoption threshold
        self.econ_threshold = stats.uniform(loc=0.80, scale=0.10)
        
        # Market adoption lag (years)
        self.market_lag = stats.expon(scale=0.5)
        
        # Measurement error
        self.measurement_error = stats.norm(loc=0, scale=0.02)
    
    def sample_parameters(self, n_samples):
        """Generate parameter samples for Monte Carlo simulation"""
        return {
            'quality_rate': self.quality_rate.rvs(n_samples),
            'econ_threshold': self.econ_threshold.rvs(n_samples),
            'market_lag': self.market_lag.rvs(n_samples),
            'measurement_error': self.measurement_error.rvs(n_samples)
        }
```

### A4.2 Simulation Engine

```python
def monte_carlo_simulation(n_simulations=10000, time_horizon=10):
    """
    Monte Carlo simulation for timeline prediction
    """
    param_dist = ParameterDistributions()
    results = []
    
    for i in range(n_simulations):
        # Sample uncertain parameters
        params = param_dist.sample_parameters(1)
        
        # Simulate quality trajectory
        timeline = simulate_trajectory(params, time_horizon)
        
        # Find threshold crossing times
        tech_threshold_time = find_threshold_crossing(timeline, 0.90)
        econ_threshold_time = find_threshold_crossing(timeline, params['econ_threshold'][0])
        
        # Calculate market displacement time
        displacement_time = econ_threshold_time + params['market_lag'][0]
        
        # Store results
        results.append({
            'simulation_id': i,
            'tech_threshold_time': tech_threshold_time,
            'econ_threshold_time': econ_threshold_time,
            'displacement_time': displacement_time,
            'quality_rate': params['quality_rate'][0],
            'econ_threshold': params['econ_threshold'][0],
            'market_lag': params['market_lag'][0]
        })
    
    return pd.DataFrame(results)

def analyze_simulation_results(results_df):
    """
    Analyze Monte Carlo simulation results
    """
    analysis = {}
    
    # Timeline statistics
    for metric in ['tech_threshold_time', 'displacement_time']:
        analysis[metric] = {
            'mean': results_df[metric].mean(),
            'median': results_df[metric].median(),
            'std': results_df[metric].std(),
            'ci_95': [
                results_df[metric].quantile(0.025),
                results_df[metric].quantile(0.975)
            ],
            'prob_before_2026': (results_df[metric] < 6.0).mean()
        }
    
    # Sensitivity analysis
    sensitivity = {}
    for param in ['quality_rate', 'econ_threshold', 'market_lag']:
        correlation = results_df[param].corr(results_df['displacement_time'])
        sensitivity[param] = correlation
    
    analysis['sensitivity'] = sensitivity
    
    return analysis
```

### A4.3 Validation and Robustness Testing

**Cross-Validation Framework:**
```python
def time_series_cross_validation(historical_data, model_func, n_folds=5):
    """
    Perform time series cross-validation for model validation
    """
    n_obs = len(historical_data)
    fold_size = n_obs // n_folds
    errors = []
    
    for fold in range(n_folds):
        # Define train/test split with expanding window
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n_obs)
        
        train_data = historical_data[:train_end]
        test_data = historical_data[test_start:test_end]
        
        # Fit model on training data
        model = model_func(train_data)
        
        # Make predictions on test data
        predictions = model.predict(len(test_data))
        
        # Calculate error metrics
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        rmse = np.sqrt(np.mean((test_data - predictions)**2))
        
        errors.append({'fold': fold, 'mape': mape, 'rmse': rmse})
    
    return pd.DataFrame(errors)

def sensitivity_analysis(base_params, param_ranges, n_points=20):
    """
    Perform systematic sensitivity analysis
    """
    results = {}
    
    for param_name, (min_val, max_val) in param_ranges.items():
        param_values = np.linspace(min_val, max_val, n_points)
        timelines = []
        
        for param_val in param_values:
            # Create modified parameters
            modified_params = base_params.copy()
            modified_params[param_name] = param_val
            
            # Run simulation with modified parameters
            timeline = predict_timeline(modified_params)
            timelines.append(timeline)
        
        results[param_name] = {
            'param_values': param_values,
            'predicted_timelines': timelines,
            'sensitivity_ratio': np.std(timelines) / np.std(param_values)
        }
    
    return results
```

## A5. Economic Modeling Framework

### A5.1 Cost Structure Analysis

**Human Translation Cost Components:**
```python
class HumanTranslationCosts:
    """
    Model human translation cost structure
    """
    
    def __init__(self):
        # Base rates (USD per 1000 words)
        self.translator_rate = 180  # Professional translator rate
        self.reviewer_rate = 80     # Quality reviewer rate
        self.pm_rate = 40          # Project management overhead
        
        # Time factors (hours per 1000 words)
        self.translation_time = 4.0
        self.review_time = 1.5
        self.pm_time = 0.5
        
        # Quality factors
        self.revision_probability = 0.15  # Probability of requiring revision
        self.revision_time_factor = 0.3   # Additional time for revisions
    
    def calculate_total_cost(self, word_count):
        """Calculate total cost for human translation project"""
        base_cost = (
            self.translator_rate + 
            self.reviewer_rate + 
            self.pm_rate
        ) * (word_count / 1000)
        
        # Add revision costs
        revision_cost = base_cost * self.revision_probability * self.revision_time_factor
        
        total_cost = base_cost + revision_cost
        total_time = (
            self.translation_time + 
            self.review_time + 
            self.pm_time
        ) * (word_count / 1000)
        
        return {
            'total_cost': total_cost,
            'total_time': total_time,
            'cost_per_word': total_cost / word_count,
            'words_per_hour': word_count / total_time
        }
```

**LLM Translation Cost Components:**
```python
class LLMTranslationCosts:
    """
    Model LLM translation cost structure
    """
    
    def __init__(self, model='gpt-4'):
        # API costs (USD per 1000 tokens)
        self.api_costs = {
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'claude-3': {'input': 0.015, 'output': 0.075},
            'gemini-pro': {'input': 0.001, 'output': 0.002}
        }
        
        self.model = model
        self.tokens_per_word = 1.3  # Approximate ratio for Swedish/English
        
        # Post-editing costs
        self.post_edit_rate = 60    # USD per hour
        self.post_edit_time_factor = 0.3  # Hours per 1000 words
        
        # Quality assurance
        self.qa_rate = 80
        self.qa_time_factor = 0.2
    
    def calculate_total_cost(self, word_count, post_editing=True):
        """Calculate total cost for LLM translation project"""
        # API costs
        token_count = word_count * self.tokens_per_word
        input_cost = (token_count / 1000) * self.api_costs[self.model]['input']
        output_cost = (token_count / 1000) * self.api_costs[self.model]['output']
        api_cost = input_cost + output_cost
        
        # Post-editing costs (if applicable)
        if post_editing:
            pe_cost = self.post_edit_rate * self.post_edit_time_factor * (word_count / 1000)
            qa_cost = self.qa_rate * self.qa_time_factor * (word_count / 1000)
        else:
            pe_cost = qa_cost = 0
        
        total_cost = api_cost + pe_cost + qa_cost
        total_time = (self.post_edit_time_factor + self.qa_time_factor) * (word_count / 1000)
        
        return {
            'api_cost': api_cost,
            'post_editing_cost': pe_cost,
            'qa_cost': qa_cost,
            'total_cost': total_cost,
            'total_time': total_time,
            'cost_per_word': total_cost / word_count,
            'processing_time': 0.1  # Minutes for API processing
        }
```

### A5.2 Economic Threshold Modeling

```python
def economic_threshold_analysis(quality_levels, cost_structures):
    """
    Analyze economic thresholds for LLM adoption
    """
    results = []
    
    for quality in quality_levels:
        for cost_scenario in cost_structures:
            # Calculate cost efficiency for humans
            human_efficiency = quality / cost_scenario['human_cost_per_word']
            
            # Calculate cost efficiency for LLMs at given quality level
            llm_cost = cost_scenario['llm_base_cost'] * (1 - cost_scenario['quality_adjustment'] * quality)
            llm_efficiency = quality / llm_cost
            
            # Determine adoption probability
            efficiency_ratio = llm_efficiency / human_efficiency
            adoption_prob = 1 / (1 + np.exp(-(efficiency_ratio - 1) / 0.2))
            
            results.append({
                'quality_level': quality,
                'cost_scenario': cost_scenario['name'],
                'human_efficiency': human_efficiency,
                'llm_efficiency': llm_efficiency,
                'efficiency_ratio': efficiency_ratio,
                'adoption_probability': adoption_prob
            })
    
    return pd.DataFrame(results)
```

## A6. Data Quality and Validation Procedures

### A6.1 Dataset Curation Protocol

**Source Text Selection:**
1. **Corpus Sampling**: Stratified random sampling from validated parallel corpora
2. **Length Distribution**: Controlled for sentence length (10-50 words)
3. **Complexity Assessment**: Flesch Reading Ease scores for English, LIX for Swedish
4. **Domain Balance**: Predetermined distribution across content types
5. **Duplicate Detection**: Shingling and hash-based deduplication

**Quality Assurance:**
```python
def validate_dataset_quality(dataset):
    """
    A dataset quality validation
    """
    quality_metrics = {}
    
    # Length distribution analysis
    lengths = [len(sentence.split()) for sentence in dataset]
    quality_metrics['length_stats'] = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': min(lengths),
        'max': max(lengths)
    }
    
    # Language detection validation
    detected_languages = [detect_language(sent) for sent in dataset]
    quality_metrics['language_accuracy'] = calculate_language_accuracy(detected_languages)
    
    # Duplicate detection
    quality_metrics['duplicate_rate'] = detect_duplicates(dataset)
    
    # Complexity assessment
    complexity_scores = [calculate_complexity(sent) for sent in dataset]
    quality_metrics['complexity_distribution'] = np.histogram(complexity_scores, bins=10)
    
    return quality_metrics

def detect_duplicates(sentences, threshold=0.8):
    """
    Detect near-duplicate sentences using semantic similarity
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = model.encode(sentences)
    
    duplicates = 0
    total_pairs = len(sentences) * (len(sentences) - 1) // 2
    
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
            if similarity > threshold:
                duplicates += 1
    
    return duplicates / total_pairs
```

### A6.2 Evaluation Consistency Protocols

**Temporal Consistency:**
- Re-evaluation of subset (n=100) at different time points
- Consistency threshold: <5% variation in scores
- Documentation of any systematic changes

**Cross-Platform Validation:**
- Parallel evaluation on different computational environments
- API version control and parameter documentation
- Reproducibility testing with identical inputs

**Blind Evaluation Protocols:**
- Randomization of evaluation order
- Anonymization of system outputs
- Independent evaluation by multiple assessors

## A7. Ethical Considerations and IRB Protocol

### A7.1 Human Subjects Protection

**IRB Approval Process:**
1. Protocol submission to Institutional Review Board
2. Consent form development and approval
3. Risk assessment and mitigation strategies
4. Data protection and privacy protocols

**Informed Consent Elements:**
- Study purpose and procedures
- Time commitment and compensation
- Data usage and storage policies
- Right to withdraw without penalty
- Contact information for questions or concerns

**Data Protection Measures:**
- Anonymization of all personal identifiers
- Secure storage with encrypted databases
- Limited access authorization
- Data retention and disposal policies

### A7.2 Research Integrity

**Reproducibility Standards:**
- Complete methodology documentation
- Code and data availability (where legally permissible)
- Version control for all analysis scripts
- Computational environment specification

**Bias Mitigation:**
- Blind evaluation protocols
- Systematic randomization procedures
- Pre-registration of hypotheses and analysis plans
- Sensitivity analysis for key assumptions

**Conflict of Interest Management:**
- Disclosure of funding sources
- Declaration of commercial relationships
- Independent validation by external collaborators
- Transparent reporting of limitations and uncertainties

---

*Methodology and Research Protocols completed: August 19, 2025*
*Total methodology documentation: 8,000+ words*
