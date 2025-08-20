# Appendix B: Mathematical Formulas and Evaluation Metrics

## B.1 Translation Quality Evaluation Metrics

### B.1.1 BLEU Score (Bilingual Evaluation Understudy)

**Formula:**
```
BLEU = BP × exp(∑(n=1 to N) w_n × log p_n)
```

Where:
- **BP (Brevity Penalty)**: `BP = min(1, exp(1 - r/c))`
  - r = reference length
  - c = candidate translation length
- **p_n**: n-gram precision = `(∑ Count_clip(n-gram)) / (∑ Count(n-gram))`
- **w_n**: weight for n-gram (typically 1/4 for n=1,2,3,4)

**Implementation for Swedish-English:**
```python
def bleu_score(candidate, reference, max_n=4):
    weights = [0.25] * max_n
    precisions = []
    
    for n in range(1, max_n + 1):
        p_n = modified_precision(candidate, reference, n)
        precisions.append(p_n)
    
    bp = brevity_penalty(candidate, reference)
    score = bp * exp(sum(w * log(p) for w, p in zip(weights, precisions)))
    return score
```

### B.1.2 ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation)

**Formula:**
```
ROUGE-L = F_lcs = ((1 + β²) × R_lcs × P_lcs) / (R_lcs + β² × P_lcs)
```

Where:
- **R_lcs**: Recall = LCS(X,Y) / m (m = length of reference)
- **P_lcs**: Precision = LCS(X,Y) / n (n = length of candidate)
- **LCS(X,Y)**: Longest Common Subsequence between candidate and reference
- **β**: Parameter controlling relative importance (typically β² = 1)

### B.1.3 METEOR (Metric for Evaluation of Translation with Explicit ORdering)

**Formula:**
```
METEOR = (1 - γ × (chunks/unigrams_matched)^θ) × F_mean
```

Where:
- **F_mean**: Harmonic mean of unigram precision and recall
- **chunks**: Number of adjacent unigrams in candidate that appear in reference
- **γ, θ**: Parameters (typically γ=0.5, θ=3)

**F_mean calculation:**
```
F_mean = (10 × P × R) / (R + 9 × P)
```

### B.1.4 BERTScore

**Formula:**
```
BERTScore_F1 = (2 × Precision × Recall) / (Precision + Recall)
```

Where:
```
Precision = (1/|x̂|) × ∑(i∈x̂) max(j∈x) cos(ê_i, e_j)
Recall = (1/|x|) × ∑(j∈x) max(i∈x̂) cos(ê_i, e_j)
```

- **x̂**: Candidate translation tokens
- **x**: Reference translation tokens
- **ê_i, e_j**: BERT embeddings of tokens
- **cos**: Cosine similarity

### B.1.5 GEMBA (GPT Estimation Metric Based Assessment)

**Scoring Function:**
```
GEMBA_Score = LLM_Evaluate(source_text, candidate_translation, reference_translation)
```

**Prompt Template:**
```
Score the following translation from Swedish to English on a scale of 1-5:

Source (Swedish): {source_text}
Translation: {candidate_translation}
Reference: {reference_translation}

Consider:
- Accuracy of meaning preservation
- Fluency and naturalness
- Cultural appropriateness
- Terminology consistency

Score: [1-5]
```

## B.2 Performance Trajectory Mathematical Models

### B.2.1 Logistic Growth Model

**Base Formula:**
```
Quality(t) = Q_max / (1 + e^(-k(t-t_0)))
```

**Parameter Estimation (Swedish-English):**
- **Q_max = 0.95**: Maximum achievable quality (95% human performance)
- **k = 0.45**: Growth rate parameter
- **t_0 = 4.2**: Inflection point (year 2024.2)

**Fitted Model:**
```
Quality(t) = 0.95 / (1 + e^(-0.45(t-4.2)))
```

**Derivative (Rate of Change):**
```
dQ/dt = (k × Q_max × e^(-k(t-t_0))) / (1 + e^(-k(t-t_0)))²
```

### B.2.2 Gompertz Growth Model

**Formula:**
```
Quality(t) = Q_max × e^(-e^(-k(t-t_0)))
```

**Fitted Parameters:**
- **Q_max = 0.96**
- **k = 0.52**
- **t_0 = 3.8**

**Swedish-English Model:**
```
Quality(t) = 0.96 × e^(-e^(-0.52(t-3.8)))
```

### B.2.3 Power Law Model

**Formula:**
```
Quality(t) = Q_0 × t^α
```

Where:
- **Q_0**: Initial quality baseline
- **α**: Scaling exponent (derived empirically)

**Empirical Estimation:**
```
α = log(Q_final/Q_initial) / log(t_final/t_initial)
```

## B.3 Economic Modeling Framework

### B.3.1 Cost Efficiency Model

**Formula:**
```
CE(t) = (Quality(t) × Speed(t)) / Cost(t)
```

**Components:**
- **Quality(t)**: From performance trajectory models
- **Speed(t)**: Processing speed improvement over time
- **Cost(t)**: Operational cost per unit of translation

**Speed Improvement Model:**
```
Speed(t) = Speed_0 × (1 + r_speed)^t
```

**Cost Reduction Model:**
```
Cost(t) = Cost_0 × (1 - r_cost)^t
```

### B.3.2 Adoption Threshold Model

**Formula:**
```
Adoption_Probability(t) = 1 / (1 + e^(-(CE(t) - CE_threshold)/σ))
```

Where:
- **CE_threshold**: Economic adoption threshold
- **σ**: Market sensitivity parameter
- **CE(t)**: Cost efficiency at time t

**Market Penetration Model:**
```
Market_Share(t) = M_max × Adoption_Probability(t)
```

### B.3.3 Total Cost of Ownership (TCO) Comparison

**Human Translation TCO:**
```
TCO_human = Labor_Cost + Management_Cost + QA_Cost + Infrastructure_Cost
```

**LLM Translation TCO:**
```
TCO_LLM = API_Cost + Post_Edit_Cost + Infrastructure_Cost + Technology_License
```

**Break-Even Analysis:**
```
Break_Even_Point = t where TCO_LLM(t) = TCO_human(t)
```

## B.4 Monte Carlo Simulation Framework

### B.4.1 Parameter Distributions

**Quality Improvement Rate:**
```
r_quality ~ N(μ=0.15, σ=0.05)  # Normal distribution
```

**Economic Threshold:**
```
threshold ~ U(0.80, 0.90)  # Uniform distribution
```

**Market Adoption Lag:**
```
lag ~ Exp(λ=0.5)  # Exponential distribution
```

**Measurement Error:**
```
error ~ N(μ=0, σ=0.05)  # Normal distribution
```

### B.4.2 Simulation Algorithm

```python
def monte_carlo_simulation(n_simulations=10000):
    results = []
    
    for i in range(n_simulations):
        # Sample parameters
        r_qual = np.random.normal(0.15, 0.05)
        threshold = np.random.uniform(0.80, 0.90)
        lag = np.random.exponential(0.5)
        
        # Simulate quality trajectory
        timeline = simulate_quality_trajectory(r_qual)
        
        # Find threshold crossing
        threshold_time = find_threshold_crossing(timeline, threshold)
        
        # Add market lag
        displacement_time = threshold_time + lag
        
        results.append({
            'threshold_time': threshold_time,
            'displacement_time': displacement_time,
            'quality_rate': r_qual,
            'threshold': threshold
        })
    
    return results
```

### B.4.3 Statistical Analysis

**Confidence Intervals:**
```
CI_lower = percentile(results, (100-confidence)/2)
CI_upper = percentile(results, (100+confidence)/2)
```

**Sensitivity Analysis:**
```
sensitivity = ∂(displacement_time) / ∂(parameter) × (parameter/displacement_time)
```

## B.5 Timeline Prediction Formulas

### B.5.1 Technical Threshold Achievement

**90% Human Performance Probability:**
```
P(Quality(t) ≥ 0.9) = ∫[0.9 to ∞] f_quality(q,t) dq
```

**Expected Timeline:**
```
E[T_90%] = ∫[0 to ∞] t × f_threshold(t) dt
```

### B.5.2 Market Displacement Timeline

**Market Penetration Rate:**
```
dM/dt = r × M(t) × (1 - M(t)/M_max)
```

**Solution:**
```
M(t) = M_max / (1 + ((M_max - M_0)/M_0) × e^(-rt))
```

### 5.3 Uncertainty Quantification

**Variance of Timeline Prediction:**
```
Var[T] = E[T²] - (E[T])²
```

**Standard Error:**
```
SE[T] = √(Var[T]/n)
```

**95% Confidence Interval:**
```
CI_95% = E[T] ± 1.96 × SE[T]
```

## 6. Domain-Specific Adjustment Factors

### 6.1 Domain Complexity Weighting

**Formula:**
```
Quality_domain(t) = Quality_general(t) × Domain_Factor × Complexity_Penalty
```

**Domain Factors:**
- General text: 1.0
- News/Media: 0.95
- Technical: 0.85
- Legal: 0.65
- Medical: 0.70
- Literary: 0.55

### 6.2 Specialized Vocabulary Impact

**Terminology Accuracy Model:**
```
Term_Accuracy = Base_Accuracy × (1 - Specialized_Ratio × Difficulty_Factor)
```

## 7. Model Validation Metrics

### 7.1 Forecasting Accuracy

**Mean Absolute Percentage Error (MAPE):**
```
MAPE = (100/n) × ∑|((Actual_i - Predicted_i)/Actual_i)|
```

**Root Mean Square Error (RMSE):**
```
RMSE = √((1/n) × ∑(Actual_i - Predicted_i)²)
```

### 7.2 Direction Accuracy

**Formula:**
```
Direction_Accuracy = (Number_of_Correct_Trend_Predictions) / (Total_Predictions)
```

### 7.3 Model Selection Criteria

**Akaike Information Criterion (AIC):**
```
AIC = 2k - 2ln(L)
```

Where:
- k = number of parameters
- L = maximum likelihood

**Bayesian Information Criterion (BIC):**
```
BIC = k×ln(n) - 2ln(L)
```

Where n = number of observations

## 8. Risk Assessment Formulas

### 8.1 Value at Risk (VaR) for Timeline

**Formula:**
```
VaR_α = inf{t : P(T ≤ t) ≥ α}
```

For α = 0.05 (95% confidence):
```
VaR_0.05 = 5th percentile of timeline distribution
```

### 8.2 Expected Shortfall

**Formula:**
```
ES_α = E[T | T ≤ VaR_α]
```

### 8.3 Scenario Probability Weighting

**Optimistic Scenario Weight:**
```
w_opt = P(Quality_Rate > μ + σ) × P(Threshold < μ_threshold)
```

**Conservative Scenario Weight:**
```
w_cons = P(Quality_Rate < μ - σ) × P(Threshold > μ_threshold)
```

## 9. Implementation Code Examples

### 9.1 Quality Trajectory Modeling

```python
import numpy as np
from scipy.optimize import curve_fit

def logistic_model(t, Q_max, k, t_0):
    return Q_max / (1 + np.exp(-k * (t - t_0)))

def fit_trajectory_model(time_data, quality_data):
    popt, pcov = curve_fit(logistic_model, time_data, quality_data)
    Q_max, k, t_0 = popt
    return Q_max, k, t_0, pcov

def predict_timeline(Q_max, k, t_0, target_quality=0.9):
    # Solve for t when Quality(t) = target_quality
    t_target = t_0 - (1/k) * np.log((Q_max/target_quality) - 1)
    return t_target
```

### 9.2 Monte Carlo Simulation

```python
def run_monte_carlo(n_sims=10000, time_horizon=10):
    results = []
    
    for _ in range(n_sims):
        # Sample uncertain parameters
        k = np.random.normal(0.45, 0.1)
        Q_max = np.random.normal(0.95, 0.02)
        threshold = np.random.uniform(0.85, 0.95)
        
        # Calculate timeline
        t_0 = 4.2  # Fixed inflection point
        timeline = predict_timeline(Q_max, k, t_0, threshold)
        results.append(timeline)
    
    return np.array(results)

def analyze_results(results):
    mean_time = np.mean(results)
    std_time = np.std(results)
    ci_lower = np.percentile(results, 2.5)
    ci_upper = np.percentile(results, 97.5)
    
    return {
        'mean': mean_time,
        'std': std_time,
        'ci_95': (ci_lower, ci_upper),
        'prob_before_2026': np.mean(results < 6.0)  # 2026 relative to 2020
    }
```

### 9.3 Economic Modeling

```python
def economic_model(t, quality_t, speed_factor=1.2, cost_reduction=0.15):
    # Quality from trajectory model
    speed_t = speed_factor ** t
    cost_t = 1 * (1 - cost_reduction) ** t
    
    cost_efficiency = (quality_t * speed_t) / cost_t
    return cost_efficiency

def adoption_probability(cost_efficiency, threshold=2.0, sensitivity=0.5):
    return 1 / (1 + np.exp(-(cost_efficiency - threshold) / sensitivity))
```

## 10. Validation and Testing Framework

### 10.1 Cross-Validation

```python
def time_series_cv(data, model_func, n_splits=5):
    n = len(data)
    errors = []
    
    for i in range(n_splits):
        # Expanding window approach
        train_size = int(n * (0.6 + 0.08 * i))
        train_data = data[:train_size]
        test_data = data[train_size:train_size+12]  # 12-month forecast
        
        model = model_func(train_data)
        predictions = model.predict(len(test_data))
        
        error = np.mean(np.abs(predictions - test_data))
        errors.append(error)
    
    return np.mean(errors), np.std(errors)
```

### 10.2 Model Comparison

```python
def compare_models(data, models):
    results = {}
    
    for name, model_func in models.items():
        mape, rmse = evaluate_model(data, model_func)
        aic = calculate_aic(data, model_func)
        bic = calculate_bic(data, model_func)
        
        results[name] = {
            'MAPE': mape,
            'RMSE': rmse,
            'AIC': aic,
            'BIC': bic
        }
    
    return results
```

This collection of mathematical formulas provides the quantitative foundation for predicting LLM displacement timelines in translation services, with specific focus on Swedish-English language pairs but generalizable to other contexts. These formulas are referenced throughout "Forecasting the Timeline of Shift to Large Language Models in Text-to-Text Translation.md" and research methodology.
