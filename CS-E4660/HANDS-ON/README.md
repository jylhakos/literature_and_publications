# End-to-End Machine Learning (ML) Systems Development

## Introduction

End-to-End ML Systems development includes many phases, such as data collection, data pre-processing, ML model serving, and (training of) the ML models. This tutorial covers the lifecycle of machine learning systems from data acquisition to model deployment and monitoring.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Setup Instructions](#setup-instructions)
3. [Questions and Answers](#questions-and-answers)
4. [Case Study: BTS Prediction](#case-study-bts-prediction)
5. [Python Files Overview](#python-files-overview)
6. [References](#references)

## Prerequisites

- Python 3
- Python 3 Virtualenv
- Pandas
- Numpy
- TensorFlow
- MLflow
- scikit-learn
- Jupyter Notebook

## Setup Instructions

### 1. Install Python Virtual Environment
```bash
sudo apt install python3 python3-venv
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install mlflow
pip install scikit-learn
pip install jupyter
pip install pandas numpy tensorflow
```

### 4. Register Jupyter Kernel
```bash
python -m ipykernel install --user --name=ml_systems_env
```
---

## Questions and Answers

### Question 1: How can you effectively manage the end-to-end process from having a dataset to serving prediction models?

**Answer 1:**

Effective management of the end-to-end ML process requires a structured approach that encompasses:

1. **Requirements Analysis**: Start by defining business requirements, performance metrics, and constraints
2. **Data Management**: Establish proper data governance, including data quality assessment, versioning, and metadata management
3. **Model Development Pipeline**: Implement systematic model development with experiment tracking, version control, and reproducible environments
4. **Deployment Strategy**: Use containerization and orchestration tools for scalable model serving
5. **Monitoring and Maintenance**: Continuous monitoring of model performance, data drift, and system health
6. **Lifecycle Management**: Establish processes for model updates, rollbacks, and retirement

Key components include:

- Data lineage tracking
- Experiment management (using tools like MLflow)
- Model registry and versioning
- Automated testing and validation
- Infrastructure as Code (IaC)
- Performance monitoring and alerting

### Question 2: What are key steps in preparing data?

**Answer 2:**

Data preparation involves several steps to transform raw data into ML-ready format:

1. **Data Investigation and Understanding**:
   - Explore data structure, types, and distributions
   - Identify patterns, correlations, and anomalies
   - Understand business context and domain knowledge

2. **Data Cleaning**:
   - **Missing Value Handling**: Use replacing, removal, or flagging strategies
   - **Outlier Detection and Treatment**: Statistical methods or domain expertise
   - **Noise Filtering**: Remove or correct erroneous data points
   - **Duplicate Removal**: Identify and handle duplicate records

3. **Data Transformation**:
   - **Normalization/Standardization**: Scale features to similar ranges
   - **Encoding**: Convert categorical variables to numerical format
   - **Feature Engineering**: Create new features from existing ones
   - **Time Series Processing**: Handle temporal dependencies and seasonality

4. **Data Validation**:
   - Schema validation
   - Data quality checks
   - Consistency verification across different data sources

As noted in the tutorial: "In practice, raw datasets are not often suitable for machine learning techniques. Most machine learning techniques are accompanied by a data-cleaning process which includes identifying outliers, filtering out noises, and handling missing values."

### Question 3: How/which ML model can be applied?

**Answer 3:**

The choice of ML model depends on several factors:

**For Time Series Prediction (like BTS data)**:
- **LSTM (Long Short-Term Memory)**: Excellent for sequential data with long-term dependencies
  - Handles vanishing gradient problem in traditional RNNs
  - Suitable for time series forecasting, speech recognition, and sequence modeling
  - Can process arbitrary length sequences and maintain memory over time

**Model Selection Criteria**:
1. **Data Type**: Time series, tabular, text, images
2. **Problem Type**: Classification, regression, clustering
3. **Data Size**: Small datasets may benefit from simpler models
4. **Interpretability Requirements**: Linear models vs. black-box models
5. **Performance Requirements**: Accuracy vs. speed trade-offs
6. **Resource Constraints**: Memory and computational limitations

**For BTS Prediction Case Study**:
- LSTM is chosen for its ability to capture temporal patterns in sensor data
- The model processes sequences of 6 time steps to predict future values
- Suitable for predictive maintenance scenarios

### Question 4: What are the requirements for your ML model/service?

**Answer 4:**

ML model/service requirements encompass multiple dimensions:

**Functional Requirements**:
- **Accuracy**: Meet specified performance metrics (RMSE, precision, recall)
- **Latency**: Response time constraints (real-time vs. batch processing)
- **Throughput**: Handle expected request volume
- **Input/Output Format**: API specification and data schemas

**Non-Functional Requirements**:
- **Scalability**: Horizontal and vertical scaling capabilities
- **Availability**: Uptime guarantees (99.9%, 99.99%)
- **Reliability**: Error handling and fault tolerance
- **Security**: Authentication, authorization, and data protection
- **Monitoring**: Logging, metrics, and alerting capabilities

**Infrastructure Requirements**:
- **Compute Resources**: CPU, GPU, memory specifications
- **Storage**: Model artifacts, logs, and data storage
- **Network**: Bandwidth and connectivity requirements
- **Deployment Platform**: Cloud, on-premises, or edge deployment

**Compliance and Governance**:
- **Data Privacy**: GDPR, HIPAA compliance
- **Model Explainability**: Regulatory requirements for interpretability
- **Audit Trail**: Model versioning and change tracking

### Question 5: Why do we need to manage metadata? Which metadata of data/model must be managed?

**Answer 5:**

Metadata management is essential for maintaining transparency, reproducibility, and governance in ML systems:

**Why Metadata is Essential**:
1. **Reproducibility**: Enable exact reproduction of experiments and results
2. **Compliance**: Meet regulatory requirements and audit trails
3. **Debugging**: Trace issues back to data or model problems
4. **Collaboration**: Share context and knowledge across teams
5. **Model Governance**: Track model lineage and impact assessment

**Metadata Categories**:

**Data Metadata**:
- **Basic Information**: Dataset ID, name, version, creation/modification timestamps
- **Source Information**: Provider, URL, collection method
- **Quality Metrics**: Completeness ratio, label quality, data distribution statistics
- **Schema**: Data types, column descriptions, constraints
- **Lineage**: Data dependencies and transformation history
- **Privacy**: Sensitivity classification, retention policies

**Model Metadata**:
- **Training Information**: Algorithm, hyperparameters, training duration
- **Performance Metrics**: Accuracy, precision, recall, AUC, loss values
- **Data Dependencies**: Training/validation/test datasets used
- **Infrastructure**: Hardware specifications, software versions
- **Deployment History**: Version history, rollback information
- **Business Context**: Use case, stakeholders, success criteria

**Example from Tutorial**:
```json
{
    "Dataset_id": "bts1",
    "Name": "bts_param",
    "Version": "1.0",
    "URL": "examples/BTS_Data/raw_data",
    "Size": {"value": "194,5", "unit": "MB"},
    "Data_type": ["Time Series"],
    "Quality_of_data": {
        "completeness": {"Value": "0.9"},
        "label_ratio": {"Value": "0.9"}
    }
}
```

### Question 6: Dealing with dynamic changes in data quality, how can we improve model serving?

**Answer 6:**

Dynamic data quality changes require proactive monitoring and adaptive strategies:

**Detection Strategies**:
1. **Data Drift Monitoring**: Statistical tests to detect distribution changes
2. **Feature Drift Detection**: Monitor individual feature distributions
3. **Performance Degradation Tracking**: Compare current vs. historical performance
4. **Real-time Quality Metrics**: Continuous assessment of data completeness, consistency

**Improvement Approaches**:

**Adaptive Models**:
- **Online Learning**: Models that update incrementally with new data
- **Ensemble Methods**: Combine multiple models for robustness
- **Multi-model Serving**: A/B testing between model versions

**Infrastructure Solutions**:
- **Data Validation Pipelines**: Automated quality checks before model inference
- **Fallback Mechanisms**: Default predictions when data quality is poor
- **Model Retraining Triggers**: Automatic retraining when performance degrades
- **Feature Store**: Centralized, validated feature repository

**Monitoring and Alerting**:
- **Dashboard Visualization**: Real-time data quality metrics
- **Automated Alerts**: Threshold-based notifications for quality issues
- **Feedback Loops**: Incorporate user feedback to improve model performance

**Quality Assurance Framework**:
- **Data Lineage Tracking**: Understand data flow and transformation points
- **Canary Deployments**: Gradual rollout of model updates
- **Shadow Mode Testing**: Compare new models against production without impact

---

## Case Study: BTS Prediction

This tutorial includes a case study of ML development for predictive maintenance in BTS (Base Transceiver Stations).

### Data Understanding and Characterization

#### Question 7: Do you understand the data?

**Answer 7:**

The BTS (Base Transceiver Station) data represents monitoring information from telecommunications infrastructure:

- **Data Type**: Time series sensor data from power grid monitoring
- **Structure**: Contains station_id, parameter_id, value, reading_time, and timestamps
- **Purpose**: Predictive maintenance to prevent equipment failures
- **Domain**: Telecommunications infrastructure monitoring
- **Temporal Nature**: Sequential measurements over time requiring temporal modeling

#### Question 8: Is the current form of data good for us to start?

**Answer 8:**

No, the raw BTS data requires significant preprocessing:

- **Raw Format Issues**: Data is in raw timestamp format requiring normalization
- **Grouping Needed**: Must be grouped by station_id and parameter_id for individual predictions
- **Temporal Structure**: Needs transformation into sequential format for LSTM processing
- **Missing Value Handling**: Requires cleaning and preprocessing steps
- **Feature Engineering**: Need to create time-based features and normalize values

#### Question 9: What does it mean "good"?

**Answer 9:**

"Good" data for ML means:

- **ML-Ready Format**: Structured, cleaned, and preprocessed
- **Appropriate Scale**: Normalized or standardized features
- **Temporal Structure**: Properly sequenced for time series analysis
- **Quality Assured**: Missing values handled, outliers addressed
- **Feature Engineered**: Relevant features extracted for the specific ML task
- **Properly Split**: Train/validation/test sets prepared
- **Documented**: Well-understood with proper metadata

#### Question 10: Which fields of data are important that can be used for ML?

**Answer 10:**

For BTS prediction, key fields include:

- **station_id**: Identifies specific BTS unit (grouping variable)
- **parameter_id**: Type of measurement (e.g., power load, temperature)
- **value**: The actual measurement value (target variable)
- **reading_time**: Timestamp for temporal ordering
- **Derived Features**:
  - **norm_time**: Normalized timestamp for temporal modeling
  - **norm_value**: Normalized measurement values
  - **Historical sequences**: Previous 6 values for LSTM input

#### Question 11: Who actually could help you to understand the data and its business?

**Answer 11:**

Domain experts who can provide context:

- **Telecommunications Engineers**: Understand BTS equipment and operations
- **Maintenance Technicians**: Know failure patterns and warning signs
- **Network Operations Teams**: Understand normal vs. abnormal behavior
- **Data Engineers**: Explain data collection and processing systems
- **Business Stakeholders**: Define success metrics and requirements
- **Subject Matter Experts**: Provide domain-specific knowledge about telecommunications infrastructure

### Data Transformation, Enrichment and Featuring

#### Question 12: Do we need to transform the data?

**Answer 12:**

Yes, significant transformation is required:

**Temporal Transformation**:
- Convert reading_time to unix timestamp
- Normalize timestamps for consistent scaling
- Create sequential windows for LSTM input

**Value Transformation**:
- Normalize measurement values by station/parameter
- Handle missing values and outliers
- Create feature sequences (6 previous values)

**Structural Transformation**:
- Group data by station_id and parameter_id
- Sort by timestamp for proper temporal order
- Split into training and testing sets

#### Question 13: Should we enrich the data with additional data? Where is the additional data?

**Answer 13:**

Additional data sources could include:

**External Data Sources**:
- **Weather Data**: Temperature, humidity affecting equipment performance
- **Historical Maintenance Records**: Previous repairs and replacements
- **Network Traffic Data**: Load patterns affecting equipment stress
- **Equipment Specifications**: Age, model, capacity information

**Derived Features**:
- **Time-based Features**: Hour of day, day of week, seasonal patterns
- **Statistical Features**: Moving averages, standard deviations
- **Lag Features**: Multiple time lags beyond the current 6-step window

**Data Location**:
- Internal maintenance systems
- Weather APIs (OpenWeather, NOAA)
- Equipment manufacturer databases
- Network monitoring systems

#### Question 14: Which features should we select for ML models and why?

**Answer 14:**

**Selected Features for LSTM Model**:

1. **Sequential Values** (norm_1 through norm_6):
   - **Why**: LSTM requires temporal sequences to learn patterns
   - **Benefit**: Captures short-term dependencies and trends

2. **Normalized Timestamps** (norm_time):
   - **Why**: Provides temporal context for predictions
   - **Benefit**: Helps model understand seasonal and cyclical patterns

3. **Station and Parameter IDs**:
   - **Why**: Different equipment may have different failure patterns
   - **Benefit**: Enables station-specific and parameter-specific modeling

**Feature Selection Criteria**:
- **Temporal Relevance**: Features that capture time-dependent patterns
- **Predictive Power**: Features correlated with equipment failures
- **Data Quality**: Features with sufficient coverage and accuracy
- **Computational Efficiency**: Balance between model complexity and performance

### Question 15: What if we don't keep track the information about the data?

**Answer 15:**

Without proper data tracking, ML systems face severe risks and limitations:

**Immediate Consequences**:
- **Reproducibility Crisis**: Unable to recreate experiments or results
- **Debugging Difficulties**: Cannot trace model failures back to data issues
- **Compliance Violations**: Fail regulatory audits and data governance requirements
- **Quality Degradation**: No visibility into data drift or quality changes

**Long-term Impact**:
- **Model Decay**: Gradual performance degradation without detection
- **Knowledge Loss**: Team changes result in lost institutional knowledge
- **Trust Erosion**: Stakeholders lose confidence in model reliability
- **Resource Waste**: Duplicate work due to lack of documentation

**Business Risks**:
- **Operational Failures**: Models fail in production without warning
- **Legal Liability**: Cannot prove compliance with data protection laws
- **Competitive Disadvantage**: Slower iteration and improvement cycles
- **Cost Escalation**: Expensive debugging and re-development efforts

As highlighted by Gebru et al. (2021) in "Datasheets for datasets," systematic documentation is essential for responsible AI development and deployment.

#### Question 16: Why do we need to keep metadata about data?

**Answer 16:**

Metadata serves as the foundation for responsible and effective ML development:

**Technical Necessity**:
1. **Reproducibility**: Enable exact replication of experiments and results
2. **Version Control**: Track data evolution and maintain compatibility
3. **Quality Assurance**: Monitor data quality metrics and detect anomalies
4. **Lineage Tracking**: Understand data flow and transformation history

**Operational Benefits**:
- **Debugging Support**: Quickly identify root causes of model issues
- **Performance Monitoring**: Track how data changes affect model performance
- **Compliance Management**: Meet regulatory requirements (GDPR, HIPAA, etc.)
- **Knowledge Sharing**: Enable effective collaboration across teams

**Strategic Value**:
- **Risk Management**: Identify and mitigate data-related risks
- **Decision Support**: Provide context for business and technical decisions
- **Innovation Enablement**: Facilitate data reuse and new model development
- **Trust Building**: Increase stakeholder confidence through transparency

#### Question 17: What are types of metadata?

**Answer 17:**

Metadata can be categorized into several essential types:

**1. Descriptive Metadata**:
- **Basic Information**: Name, version, description, creation date
- **Source Details**: Provider, collection method, geographic origin
- **Schema Information**: Data types, column descriptions, constraints
- **Business Context**: Use case, stakeholders, success criteria

**2. Administrative Metadata**:
- **Access Control**: Permissions, security classifications, privacy levels
- **Lifecycle Management**: Retention policies, archival strategies
- **Legal Information**: Licensing terms, usage restrictions, compliance status
- **Contact Information**: Data stewards, owners, technical contacts

**3. Technical Metadata**:
- **Format Details**: File formats, encoding, compression methods
- **Storage Information**: Location, backup status, access patterns
- **Processing History**: Transformations applied, validation results
- **Performance Metrics**: Query statistics, access frequency

**4. Quality Metadata**:
- **Completeness Measures**: Missing value rates, coverage statistics
- **Accuracy Indicators**: Error rates, validation results, bias metrics
- **Consistency Checks**: Duplicate detection, cross-source validation
- **Timeliness Information**: Data freshness, update frequency

**5. Lineage Metadata**:
- **Data Dependencies**: Source systems, upstream dependencies
- **Transformation History**: Processing steps, algorithm versions
- **Impact Analysis**: Downstream consumers, affected systems
- **Change Tracking**: Modification history, approval workflows

#### Question 18: How to obtain metadata?

**Answer 18:**

Metadata acquisition requires systematic approaches across multiple stages:

**1. Automated Collection**:
- **System Logs**: Extract metadata from database logs, API calls, and system events
- **Schema Discovery**: Automatically analyze data structures and relationships
- **Quality Profiling**: Use tools like Great Expectations, Apache Griffin for quality metrics
- **Lineage Tracking**: Implement tools like Apache Atlas, DataHub for data lineage

**2. Manual Documentation**:
- **Business Context**: Interview domain experts and stakeholders
- **Use Case Documentation**: Record intended purposes and success criteria
- **Quality Assessments**: Manual review and validation by data stewards
- **Compliance Information**: Legal and regulatory requirement documentation

**3. Tool-Based Approaches**:
- **Data Catalogs**: Use BigQuery Data Catalog, AWS Glue, Azure Purview
- **MLOps Platforms**: Leverage MLflow, Weights & Biases, Neptune for ML metadata
- **Data Quality Tools**: Implement Deequ, Great Expectations for quality monitoring
- **Version Control**: Use DVC, Git LFS for data versioning and tracking

**4. Integration Strategies**:
- **API Integration**: Connect metadata collection to data pipelines
- **Real-time Monitoring**: Implement streaming metadata collection
- **Batch Processing**: Regular metadata extraction and validation jobs
- **Cross-system Synchronization**: Ensure consistency across multiple systems

**5. Best Practices**:
- **Standardization**: Use common metadata schemas and vocabularies
- **Automation**: Minimize manual effort through automated collection
- **Validation**: Implement checks to ensure metadata accuracy and completeness
- **Governance**: Establish clear ownership and update responsibilities

**Example Implementation**:
```python
# Automated metadata collection example
def collect_dataset_metadata(dataset_path):
    metadata = {
        "name": os.path.basename(dataset_path),
        "size": os.path.getsize(dataset_path),
        "created_time": datetime.fromtimestamp(os.path.getctime(dataset_path)),
        "modified_time": datetime.fromtimestamp(os.path.getmtime(dataset_path)),
        "schema": infer_schema(dataset_path),
        "quality_metrics": calculate_quality_metrics(dataset_path),
        "lineage": trace_data_lineage(dataset_path)
    }
    return metadata
```

---

## Developing ML Models

### Question 19: Which ML algorithms should we choose to create suitable ML models and why?

**Answer 19:**

**LSTM (Long Short-Term Memory) is chosen for this use case because**:

**Advantages of LSTM**:
1. **Temporal Dependencies**: Excels at capturing long-term patterns in sequential data
2. **Vanishing Gradient Solution**: Overcomes traditional RNN limitations
3. **Memory Mechanism**: Can remember important information over long sequences
4. **Flexibility**: Handles variable-length sequences effectively

**Why LSTM for BTS Prediction**:
- **Time Series Nature**: BTS monitoring data is inherently temporal
- **Pattern Recognition**: Can identify patterns leading to equipment failures
- **Long-term Memory**: Remembers relevant historical conditions
- **Noise Handling**: Robust to sensor noise and missing data

**Alternative Models Considered**:
- **Traditional RNN**: Suffers from vanishing gradient problem
- **Linear Regression**: Too simple for complex temporal patterns
- **Random Forest**: Good for tabular data but not optimal for sequences
- **CNN**: Better for spatial data than temporal sequences

### Training and ML Model Experiments

#### Question 20: How will you do the training and model experiments?

**Answer 20:**

**Training Strategy**:
1. **Data Preparation**:
   - Split data into train/validation/test sets (70/15/15)
   - Create sequential windows of 6 time steps
   - Normalize features and targets

2. **Model Architecture**:
   - Multiple LSTM layers with configurable nodes
   - TimeDistributed Dense layer for output
   - Adam optimizer with learning rate 0.005

3. **Training Process**:
   - Batch size: 1 (online learning)
   - Epochs: 2 (configurable)
   - Validation monitoring for early stopping

4. **Hyperparameter Tuning**:
   - Number of LSTM layers
   - Nodes per layer
   - Learning rate optimization
   - Dropout rates for regularization

#### Question 21: How will we record performance metrics, machine information, etc. and associate them with the data to be used (and the metadata) so that we can have all information linked for an end-to-end ML experiment?

**Answer 21:**

**Using MLflow for Experiment Tracking**:

```python
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("number of layer", n_layers)
    mlflow.log_param("number of node each layer", node_config)
    
    # Log metrics
    mlflow.log_metric("loss", training_loss)
    mlflow.log_metric("val_loss", validation_loss)
    
    # Log model
    mlflow.keras.log_model(model, "LSTM_model",
                          signature=signature,
                          input_example=input_example)
```

**Tracking Includes**:
- **Model Parameters**: Architecture, hyperparameters
- **Training Metrics**: Loss, accuracy, validation scores
- **Data Information**: Dataset version, preprocessing steps
- **Environment**: Python version, library versions, hardware specs
- **Model Artifacts**: Trained model, preprocessing pipelines

**MLflow Features**:
- **Tracking**: Track experiments to store parameters and results
- **Packaging**: Package project code in reproducible form for sharing or production transfer
- **Deploying**: Manage and deploy models from various ML libraries

**Practical Implementation**:
The BTS model code can be found in `model.py`. You can log input examples for the model using `mlflow.keras.log_model` with input_example parameter.

#### Question 22: How would we manage thousands of experiments?

**Answer 22:**

**Experiment Management Strategy**:

1. **MLflow Tracking Server**:
   - Centralized experiment logging
   - Web UI for experiment comparison
   - RESTful API for programmatic access

2. **Organization Techniques**:
   - **Experiment Naming**: Consistent naming conventions
   - **Tagging**: Categorical tags for filtering
   - **Hierarchical Structure**: Project → Experiment → Runs

3. **Automation Tools**:
   - **Hyperparameter Sweeps**: Automated parameter search
   - **Pipeline Integration**: CI/CD integration
   - **Scheduled Experiments**: Regular retraining pipelines

4. **Analysis and Comparison**:
   - **Metrics Visualization**: Compare performance across runs
   - **Model Registry**: Version control for production models
   - **Automated Reports**: Regular performance summaries

### Running Model Experiments

**Basic Execution**:
```bash
# Run the model training with default parameters
python model.py

# Modify parameters and record them with MLflow
python model.py --epochs 10 --batch_size 32
```

**Parameter Modification Examples**:
You can modify different parameters such as:
- Loss function (`loss='mean_squared_error'`)
- Batch size (`batch_size=1`)
- Number of epochs (`epochs=2`)
- Test data file selection
- LSTM layer configuration
- Learning rate settings

All parameters are automatically recorded using `mlflow.log_param()`.

**MLflow UI Access**:
```bash
# Open MLflow user interface
mlflow ui

# Alternative: Start MLflow server
mlflow server -h 0.0.0.0
```

### Examining Data and Model Experiments

Once you have metadata about:
- Data used in training
- Model architecture and parameters
- Model experiments and results
- Performance metrics

The metadata captures the model in an end-to-end view, explaining relationships between data, model, and metrics obtained from experiments.

#### Question 23: How to evaluate or compare your experiments based on multiple metrics? What would be an appropriate solution?

**Answer 23:**

**Multi-metric Evaluation Framework**:

**Primary Metrics**:
- **RMSE**: Root Mean Square Error for regression accuracy
- **MAE**: Mean Absolute Error for interpretable error magnitude
- **MAPE**: Mean Absolute Percentage Error for relative performance

**Secondary Metrics**:
- **Training Time**: Model efficiency
- **Inference Latency**: Serving performance
- **Model Size**: Memory requirements

**Comparison Solutions**:
1. **MLflow UI**: Interactive comparison of experiments
2. **Custom Dashboards**: Business-specific metric visualization
3. **Automated Ranking**: Multi-objective optimization
4. **Statistical Significance Testing**: Validate performance differences

**Decision Framework**:
- **Pareto Analysis**: Trade-off between accuracy and efficiency
- **Business Impact Scoring**: Weight metrics by business value
- **Cross-validation**: Ensure robust performance estimates

---

## Model Serving/ML Service

### Question 24: How to pack and move code to serving platforms?

**Answer 24:**

**Packaging Strategy**:

1. **MLflow Model Format**:
   - Standardized model packaging
   - Includes dependencies and metadata
   - Platform-agnostic deployment

2. **Containerization**:
   - Docker containers for consistency
   - Include all dependencies and environment
   - Enable horizontal scaling

3. **Model Registry**:
   - Version control for models
   - Staging and production environments
   - Automated deployment pipelines

**Deployment Process**:
```bash
# Package model with MLflow
mlflow models build-docker -m "models:/LSTM_model/Production" -n lstm-serving

# Deploy to serving platform
mlflow models serve -m mlruns/0/project_id/artifacts/LSTM_model -p 8888
```

### Model Packaging

**MLflow Project Configuration**:
To package code using MLflow, create MLproject and description files defining execution requirements:

```yaml
# MLproject file example
name: tutorial

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      configuration_file: {type: string, default: "conf.txt"}
    command: "python model.py --conf {configuration_file}"
```

**Virtual Environment Packaging**:
You can package code in virtual environments (conda/venv) for portability and consistent execution across different environments.
```

#### Question 25: Which service platforms should we use?

**Answer 25:**

**Platform Options**:

1. **Cloud Platforms**:
   - **AWS SageMaker**: Managed ML serving with auto-scaling
   - **Google AI Platform**: Integrated with Google Cloud services
   - **Azure ML**: Enterprise-grade ML operations

2. **Containerized Solutions**:
   - **Kubernetes**: Orchestrated container deployment
   - **Docker Swarm**: Simpler container orchestration
   - **MLflow Serving**: Built-in serving capabilities

3. **Edge Deployment**:
   - **TensorFlow Lite**: Mobile and edge device deployment
   - **ONNX Runtime**: Cross-platform inference

**Selection Criteria**:
- **Scalability Requirements**: Expected traffic and growth
- **Latency Constraints**: Real-time vs. batch processing
- **Cost Considerations**: Operational expenses
- **Integration Needs**: Existing infrastructure compatibility

#### Question 26: How to deploy and manage ML services?

**Answer 26:**

**Deployment Strategy**:

1. **Infrastructure as Code**:
   - Terraform or CloudFormation templates
   - Reproducible deployment environments
   - Version-controlled infrastructure

2. **CI/CD Pipelines**:
   - Automated testing and validation
   - Staged deployment (dev → staging → production)
   - Rollback capabilities

3. **Service Management**:
   - **Load Balancing**: Distribute traffic across instances
   - **Health Checks**: Monitor service availability
   - **Auto-scaling**: Adjust capacity based on demand
   - **Logging and Monitoring**: Observability

**Management Best Practices**:
- **Blue-green Deployments**: Zero-downtime updates
- **Canary Releases**: Gradual traffic shifting
- **Feature Flags**: Control feature rollouts
- **Disaster Recovery**: Backup and recovery procedures

### Serving Models with MLflow

**Model Format**:
MLflow Model provides a standard format for packaging ML models that can be used in various downstream tools, including serving as REST API services.

**Deployment Commands**:
```bash
# Check saved models in MLflow UI
mlflow ui

# Start MLflow server
mlflow server -h 0.0.0.0

# Deploy the server using saved model
mlflow models serve -m mlruns/0/project_id/artifacts/LSTM_model -p 8888
```

**Client Integration**:
Use the provided `client.py` to send requests to the API endpoint:
```python
# Example from client.py
import requests

API_ENDPOINT = "http://0.0.0.0:8888/invocations"
param = {
  "inputs": [[[ 0.074], [-0.003], [-0.08] ,[ -0.157], [-0.235], [-0.312]]]
}
response = requests.post(url=API_ENDPOINT, json=param)
result = response.json()
print(result[0][0])
```

---

## Monitoring and Quality Assurance

### Question 27: Assume that you want to monitor more complex metrics such as cost, performance of your API functions, what are the suitable solutions?

**Answer 27:**

**Monitoring Solutions**:

1. **Application Performance Monitoring (APM)**:
   - **New Relic**: Full-stack monitoring
   - **DataDog**: Infrastructure and application metrics
   - **Prometheus + Grafana**: Open-source monitoring stack

2. **Custom Metrics Dashboard**:
   - **Cost Tracking**: Resource utilization and billing
   - **Performance Metrics**: Latency, throughput, error rates
   - **Business Metrics**: Prediction accuracy, user satisfaction

3. **Alerting Systems**:
   - **Threshold-based Alerts**: Performance degradation notifications
   - **Anomaly Detection**: Unusual pattern identification
   - **Escalation Procedures**: Incident response workflows

**Key Metrics to Monitor**:
- **Infrastructure Costs**: Compute, storage, network expenses
- **API Performance**: Response time, request rate, error rate
- **Model Performance**: Prediction accuracy, drift detection
- **User Experience**: Satisfaction scores, usage patterns

#### Question 28: Then how can you link the monitoring data of the service back to the model, model experiments, trained data, etc.?

**Answer 28:**

**Traceability Framework**:

1. **Unique Identifiers**:
   - **Model Version IDs**: Link predictions to specific model versions
   - **Experiment IDs**: Connect to training experiments
   - **Data Version Tags**: Track data lineage

2. **Metadata Propagation**:
   - **HTTP Headers**: Include model metadata in API responses
   - **Logging Context**: Structured logs with traceability information
   - **Database Relations**: Link monitoring records to model registry

3. **End-to-End Tracking**:
   - **Request Tracing**: Follow requests through the entire pipeline
   - **Model Lineage**: Track from data → training → deployment → predictions
   - **Impact Analysis**: Understand how changes affect performance

**Implementation Approach**:
```python
# Example: Add traceability to predictions
def predict(data):
    model_version = get_current_model_version()
    prediction = model.predict(data)
    
    # Log prediction with metadata
    log_prediction(
        prediction=prediction,
        model_version=model_version,
        data_hash=hash(data),
        timestamp=datetime.now()
    )
    
    return prediction
```

---

## Python Files Overview

The tutorial includes several Python files that demonstrate the end-to-end ML pipeline:

### 1. `group_data.py`
- **Purpose**: Data preprocessing and grouping
- **Functions**:
  - Load raw CSV files from multiple sources
  - Convert timestamps and normalize time features
  - Group data by station_id and parameter_id
  - Save processed data for training

### 2. `pre_processing.py`
- **Purpose**: Prepare data for LSTM model training
- **Functions**:
  - Load grouped datasets
  - Create sequential features (6 time steps)
  - Handle missing values and data cleaning
  - Normalize values for model training

### 3. `model.py`
- **Purpose**: LSTM model training and experiment tracking
- **Functions**:
  - Build configurable LSTM architecture
  - Train model with MLflow tracking
  - Log parameters, metrics, and model artifacts
  - Save trained model for serving

### 4. `data_extraction.py`
- **Purpose**: Test data preparation
- **Functions**:
  - Load separate test dataset
  - Apply same preprocessing as training data
  - Create evaluation dataset for model testing

### 5. `client.py`
- **Purpose**: Model serving client
- **Functions**:
  - Send requests to deployed model API
  - Handle prediction responses
  - Demonstrate model serving integration

---


## Practice of ML Development

### BTS Predictive Maintenance Case Study

We will carry out a  case study of ML development for predictive maintenance in BTS (Base Transceiver Stations), demonstrating the complete end-to-end ML systems development process.

**Data Sources**:
- Example raw data: `tutorials/MLProjectManagement/BTS_Example/raw_data`
- Processed dataset: [1161114002_122_norm.csv](https://github.com/rdsea/IoTCloudSamples/blob/master/MLUnits/BTSPrediction/data/1161114002_122_norm.csv)
- BTS monitoring samples: [BTS README](https://github.com/rdsea/IoTCloudSamples/blob/master/data/bts/README.md)
- Alarm data example: [alarm-2017-10-23-12-vn.csv](https://github.com/rdsea/IoTCloudSamples/blob/master/data/bts/alarm-2017-10-23-12-vn.csv)

**Data Processing Pipeline**:
1. **Raw Data Extraction**: Convert reading_time timestamps and group by station_id and parameter_id using `group_data.py`
2. **Preprocessing**: Transform data into sequential format and normalize values using `pre_processing.py`
3. **Feature Engineering**: Create time-series windows for LSTM input
4. **Model Training**: Train LSTM model with MLflow tracking using `model.py`
5. **Model Serving**: Deploy model as REST API service
6. **Client Integration**: Test predictions using `client.py`

### Holistic Explainability Requirements

As discussed in the paper by Nguyen et al. (2021) on "Holistic Explainability Requirements for End-to-End Machine Learning in IoT Cloud Systems," end-to-end ML in IoT Cloud systems involves multiple processes covering data, model, and service engineering with multiple stakeholders.

**Key Requirements**:
- **Stakeholder Involvement**: Include domain experts, data scientists, and business users
- **Process Coverage**: Address explainability across data, model, and service layers
- **Multiple Aspects**: Consider technical, business, and regulatory explainability needs

**Reference**: [Holistic Explainability Research](https://research.aalto.fi/en/publications/holistic-explainability-requirements-for-end-to-end-machine-learn)

## Additional Learning Resources

### ML Service Monitoring

**Performance Monitoring**:
- Tutorial: [Performance Monitoring](https://github.com/rdsea/sys4bigml/tree/master/tutorials/PerformanceMonitoring)
- Focus: Monitor API performance, cost, and complex metrics

**Quality of Analytics for ML**:
- Tutorial: [QoA4ML](https://github.com/rdsea/sys4bigml/tree/master/tutorials/qoa4ml)
- Focus: Analytics quality in ML systems

### Federated Learning

**Flower Framework**:
- Website: [Flower.ai](https://flower.ai/)
- Description: An approach to federated learning, analytics, and evaluation
- Tutorial: [What is Federated Learning](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html)

### MLflow Resources

**Official Documentation**:
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [MLflow Models](https://www.mlflow.org/docs/latest/models.html#models)

**Hands-on Tutorials**:
- [Elasticity and Scalability for End-to-End ML Serving](https://github.com/rdsea/sys4bigml/tree/master/tutorials/MLServing)


## Getting Started

1. **Clone or download the tutorial files**
2. **Set up virtual environment** using the instructions above
3. **Install dependencies** from the prerequisites section
4. **Explore the Python files** to understand the ML pipeline
5. **Run the data preprocessing** scripts to prepare your data
6. **Train the LSTM model** using MLflow for experiment tracking
7. **Deploy the model** for serving predictions
8. **Monitor and maintain** the deployed system


## References

1. [Holistic Explainability Requirements for End-to-End Machine Learning in IoT Cloud Systems](https://dl.acm.org/doi/pdf/10.1145/3458723)
2. [Long Short-Term Memory - Wikipedia](https://en.wikipedia.org/wiki/Long_short-term_memory)
3. [IoT Cloud Samples - BTS Prediction](https://github.com/rdsea/IoTCloudSamples/tree/master/MLUnits/BTSPrediction)
4. [BTS Prediction Data](https://github.com/rdsea/IoTCloudSamples/tree/master/MLUnits/BTSPrediction/data)
5. [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
6. [Datasheets for Datasets Paper](https://dl.acm.org/doi/10.1145/3458723) - Timnit Gebru et al.
7. [Performance Monitoring Tutorial](https://github.com/rdsea/sys4bigml/tree/master/tutorials/PerformanceMonitoring)
8. [Quality of Analytics for ML](https://github.com/rdsea/sys4bigml/tree/master/tutorials/qoa4ml)
9. [Flower - Federated Learning](https://flower.ai/)

---

