#!/bin/bash
# Setup script for MCP AI Agent Research Project

set -e  # Exit on error

echo "=================================================="
echo "MCP AI Agent Research Project - Setup"
echo "=================================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10+ is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python $PYTHON_VERSION detected${NC}"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Check for .env file
echo "Checking configuration..."
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Warning: .env file not found${NC}"
    echo "Creating .env template..."
    cat > .env << EOF
# Anthropic API Key (required for Claude SDK approach)
ANTHROPIC_API_KEY=your_api_key_here

# Optional: Other configuration
# MODEL_CACHE_DIR=./model-cache
EOF
    echo -e "${GREEN}✓ Created .env template${NC}"
    echo -e "${YELLOW}  Please edit .env and add your API keys${NC}"
else
    echo -e "${GREEN}✓ .env file exists${NC}"
fi
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p model-cache
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check GPU availability (optional)
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GPU detected: $GPU_INFO${NC}"
        echo -e "${GREEN}  Arcee Agent will use GPU acceleration${NC}"
    else
        echo -e "${YELLOW}⚠ nvidia-smi found but no GPU detected${NC}"
        echo -e "${YELLOW}  Arcee Agent will run on CPU (slower)${NC}"
    fi
else
    echo -e "${YELLOW}⚠ CUDA not detected${NC}"
    echo -e "${YELLOW}  Arcee Agent will run on CPU (slower)${NC}"
fi
echo ""

# Summary
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment:"
echo -e "   ${GREEN}source venv/bin/activate${NC}"
echo ""
echo "2. Configure your API key in .env file"
echo ""
echo "3. Run Claude SDK agent:"
echo -e "   ${GREEN}python src/claude_mcp_agent/client.py src/servers/weather_server.py${NC}"
echo ""
echo "4. Or run Arcee Agent (requires GPU recommended):"
echo -e "   ${GREEN}python src/arcee_langchain_agent/agent.py src/servers/weather_server.py${NC}"
echo ""
echo "See README.md for more information."
echo ""
