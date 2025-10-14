# Evaluation of Model Context Protocol (MCP) with AI Agents

Research project for CS-E4660: Model Context Protocol Integration with AI Agents

## Abstract

This research explores the question **How the development of artificial intelligence (AI) agents is transformed by the Model Context Protocol (MCP)?** The research employs an experimental work comparing MCP-based development and traditional API integration approaches applying the Arcee Agent model. The research tests hypotheses across dual dimensions: (1) **development practice transformation** (implementation time, code complexity, integration effort, learning curve) and (2) **agent capability enhancement** (task completion, tool success, token efficiency, response time, error recovery). Expected outcomes include 40-75% reductions in development effort metrics and 15-50% improvements in agent performance metrics. This research provides the first systematic empirical evaluation of MCP's transformative impact, informing software engineering practices and protocol adoption decisions.

**Keywords:** Model Context Protocol, MCP, AI agents, protocol standardization, development transformation, experimental evaluation

## Research Question

**Research Problem:**
How the development of artificial intelligence (AI) agents is transformed by the Model Context Protocol (MCP)?

**Primary Research Question:**
Does Model Context Protocol (MCP) integration lead to reduced development time and improved AI agent capabilities compared to traditional API integration approaches?

**Research Type:** Confirmatory with Exploratory Components (Transformation)
- Analyzes transformative impact of MCP on AI agent development
- Uses experimental solution with randomized controlled trials
- Measures development practice transformation and agent operational improvements
- Characterizes paradigm shift from custom to protocol-driven integration

**Key Variables:**
- **Independent Variable (IV):** Development Approach (MCP Adoption)
  - Control: Traditional API Integration (custom per-service integration)
  - Treatment: MCP-Based Integration (protocol-driven standardization)

- **Dependent Variables - Category A (Development Practice Transformation):**
  - Development time reduction (hours)
  - Code complexity reduction (composite index 0-100)
  - Integration effort reduction (count of steps)
 
- **Dependent Variables - Category B (Agent Capability Enhancement):**
  - Task completion rate improvement (%)
  - Tool call success rate improvement (%)
  - Response time improvement (seconds)

**Hypotheses:**

*Development Practice Transformation (H₁-H₄):*
1. MCP adoption transforms development workflows → 40-60% time reduction
2. MCP adoption simplifies code architecture → 45-60% complexity reduction
3. MCP adoption streamlines integration → 65-75% fewer configuration steps
4. MCP adoption accelerates skill acquisition → 60-70% faster proficiency

*Agent Capability Enhancement (H₅-H₉):*
5. MCP adoption improves tool communication → 15-30% higher success rates
6. MCP adoption optimizes context usage → 20-40% token reduction
7. MCP adoption increases task completion → 20-30% higher completion rates
8. MCP adoption accelerates execution → 30-50% faster response times
9. MCP adoption strengthens resilience → Improved autonomous error recovery

**Scope:**
The purpose of this research is to explore the transformative impact of MCP on AI agent development
by analyzes both the development process changes and the resulting agent capability improvements.

For detailed methodology and theoretical framework, see `assignment.txt`.

## Project Structure

```
.
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
├── assignment.txt                      # Research paper
└── src/                                # Source code
    ├── claude_mcp_agent/               # Approach A: Claude SDK + MCP
    │   ├── __init__.py
    │   └── client.py                   # Claude MCP agent
    ├── arcee_langchain_agent/          # Approach B: Arcee + LangChain
    │   ├── __init__.py
    │   ├── config.py                   # Model configuration
    │   ├── mcp_bridge.py               # MCP-LangChain bridge
    │   └── agent.py                    # Arcee agent implementation
    ├── sagemaker_mcp_agent/            # Approach C: AWS SageMaker + MCP
    │   ├── __init__.py
    │   ├── agent.py                    # SageMaker MCP agent
    │   ├── deploy_sagemaker.py         # Deployment script
    │   └── cleanup_sagemaker.py        # Cleanup script
    └── servers/                        # MCP servers
        ├── weather_server.py           # Weather information server
        └── calculator_server.py        # Calculator server
```

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# For Claude SDK approach
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Usage

### Approach A: Claude SDK with MCP

Run the Claude MCP agent with a server:

```bash
# Activate virtual environment first
source venv/bin/activate

# Run with weather server
python src/claude_mcp_agent/client.py src/servers/weather_server.py

# Run with calculator server
python src/claude_mcp_agent/client.py src/servers/calculator_server.py
```

Example interaction:
```
You: What's the weather in London?
Agent: [Calling get_current_weather with {'location': 'London'}]
      Current weather in London: 18°C, Partly cloudy...
```

### Approach B: Arcee Agent with LangChain

⚠️ **Note**: Requires GPU with CUDA support or substantial RAM for CPU inference.

```bash
# Activate virtual environment first
source venv/bin/activate

# Run with one or more servers
python src/arcee_langchain_agent/agent.py src/servers/weather_server.py

# Run with multiple servers
python src/arcee_langchain_agent/agent.py \
    src/servers/weather_server.py \
    src/servers/calculator_server.py
```

### Approach C: AWS SageMaker with MCP

⚠️ **Prerequisites**:
- AWS account with SageMaker access
- AWS credentials configured (IAM role or access keys)
- Sufficient quota for GPU instances

**Step 1: Deploy to SageMaker**

```bash
# Activate virtual environment first
source venv/bin/activate

# Deploy with default settings (ml.g5.2xlarge, 1 GPU)
python src/sagemaker_mcp_agent/deploy_sagemaker.py

# Deploy with larger instance (ml.g5.12xlarge, 4 GPUs)
python src/sagemaker_mcp_agent/deploy_sagemaker.py \
    --instance-type ml.g5.12xlarge \
    --gpus 4

# The script will output your endpoint name
# Example: Arcee-Agent-2025-01-13-10-30-00
```

**Step 2: Run the Agent**

```bash
# Set your endpoint name (from deployment output)
export SAGEMAKER_ENDPOINT=Arcee-Agent-2025-01-13-10-30-00

# Run the agent
python src/sagemaker_mcp_agent/agent.py
```

**Step 3: Cleanup Resources**

```bash
# Delete a specific endpoint
python src/sagemaker_mcp_agent/cleanup_sagemaker.py \
    --endpoint Arcee-Agent-2025-01-13-10-30-00

# List all endpoints
python src/sagemaker_mcp_agent/cleanup_sagemaker.py --list

# Delete all Arcee-Agent endpoints
python src/sagemaker_mcp_agent/cleanup_sagemaker.py --all
```

Example interaction:
```
You: What's the weather in Tokyo and calculate 15% tip on $82.50
Agent: [Calling weather_get_current_weather]
       [Calling calculator_calculate_arithmetic]
       
       The weather in Tokyo is 18°C with partly cloudy skies.
       A 15% tip on $82.50 would be $12.38.
```

**Cost Estimation**:
- ml.g5.2xlarge: ~$1.41/hour (~$1,029/month if running 24/7)
- ml.g5.12xlarge: ~$7.09/hour (~$5,176/month if running 24/7)

**⚠️ Important**: Always delete endpoints when not in use to avoid charges!

## Comparison of Approaches

| Feature | Approach A (Claude SDK) | Approach B (Arcee + LangChain) | Approach C (SageMaker + MCP) |
|---------|-------------------------|--------------------------------|------------------------------|
| **Infrastructure** | API-based (cloud) | Local deployment | AWS managed (cloud) |
| **Setup Complexity** | Low | Medium | Medium-High |
| **Cost** | Per-token pricing | Hardware + electricity | Per-hour instance pricing |
| **Scalability** | Automatic | Limited by hardware | Auto-scaling available |
| **Performance** | High | Depends on hardware | High (GPU instances) |
| **Latency** | Network + API | Local (lowest) | Network + inference |
| **Best For** | Quick prototyping | Cost-conscious, local control | Enterprise production |
| **Requirements** | API key | GPU or powerful CPU | AWS account + quota |

## Implementation Approaches

### Approach A: Claude SDK + MCP
- **Advantages**: Easy setup, excellent performance, native MCP support
- **Use Cases**: Rapid prototyping, applications needing sophisticated reasoning
- **Requirements**: Anthropic API key

### Approach B: Arcee Agent + LangChain
- **Advantages**: Lower operational cost, full model customization, local deployment
- **Use Cases**: High-volume applications, privacy-sensitive data, cost optimization
- **Requirements**: GPU with CUDA or substantial RAM

### Approach C: AWS SageMaker + MCP
- **Advantages**: Enterprise features, auto-scaling, managed infrastructure, monitoring
- **Use Cases**: Production deployments, enterprise applications, variable traffic
- **Requirements**: AWS account, SageMaker access, sufficient quota
- **Source**: Based on [arcee-ai/aws-samples](https://github.com/arcee-ai/aws-samples)

## Development Workflow

### Activate Virtual Environment

Before running Python scripts:

```bash
# Linux/macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Testing MCP Servers

Test individual MCP servers:

```bash
# Test weather server
python src/servers/weather_server.py

# Test calculator server
python src/servers/calculator_server.py
    src/servers/calculator_server.py
```

### Running MCP Servers Standalone

You can test MCP servers independently:

```bash
# Test weather server
python src/servers/weather_server.py

# Test calculator server
python src/servers/calculator_server.py
```

## Development

### Adding New MCP Servers

1. Create a new server file in `src/servers/`
2. Implement the MCP server interface:
   - `@app.list_tools()` - Define available tools
   - `@app.call_tool()` - Implement tool execution

Example template:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("my-server")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="my_tool",
            description="Description of my tool",
            inputSchema={...}
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Implementation
    pass
```

### Deactivating Virtual Environment

When done working:

```bash
deactivate
```

## Requirements

### Minimum Requirements
- Python 3.10+
- 8GB RAM (for Claude SDK approach)
- Internet connection (for API calls)

### Recommended for Arcee Agent
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- 16GB+ System RAM

## Troubleshooting

### Virtual Environment

If `python3 -m venv` fails:
```bash
# Install venv package (Ubuntu/Debian)
sudo apt-get install python3-venv

# Or use virtualenv
pip install virtualenv
virtualenv venv
```

### Import Errors

Ensure virtual environment is activated:
```bash
# Check which python is being used
which python

# Should show path to venv/bin/python
```

### GPU Memory Issues (Arcee Agent)

Use quantization to reduce memory usage:
```python
# Modify agent.py
agent = ArceeAgentWithMCP(use_quantization=True)
```

## Testing

Run basic tests:

```bash
# Test Claude client (requires API key)
python src/claude_mcp_agent/client.py src/servers/calculator_server.py

# Query: Calculate 5 + 3
# Expected: Tool call with result 8
```

## License

MIT License - see LICENSE file for details.

