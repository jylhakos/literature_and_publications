"""
Arcee LangChain Agent Package: AI Agent using Arcee Agent SLM with LangChain and MCP
"""

from .agent import ArceeAgentWithMCP, ArceeLangChainLLM
from .mcp_bridge import MCPBridge
from .config import ArceeAgentConfig

__all__ = [
    'ArceeAgentWithMCP',
    'ArceeLangChainLLM',
    'MCPBridge',
    'ArceeAgentConfig'
]
__version__ = '1.0.0'
