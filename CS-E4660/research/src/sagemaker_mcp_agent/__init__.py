"""
AWS SageMaker MCP Agent Package

This package provides integration between AWS SageMaker endpoints
and Model Context Protocol (MCP) servers for agentic AI applications.
"""

__version__ = "1.0.0"
__author__ = "CS-E4660 Course Project"

from .agent import ArceeAgentSageMakerMCP, MCPSageMakerBridge

__all__ = ["ArceeAgentSageMakerMCP", "MCPSageMakerBridge"]
