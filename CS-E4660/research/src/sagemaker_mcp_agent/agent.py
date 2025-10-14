"""
AWS SageMaker + MCP Integration for Arcee Agent

This module provides integration between AWS SageMaker endpoints running
Arcee Agent and MCP (Model Context Protocol) servers.

Architecture:
- SageMaker endpoint hosts Arcee-Agent with TGI (Text Generation Inference)
- MCP servers provide standardized tool access
- Bridge layer connects SageMaker endpoint with MCP tools
- OpenAI-compatible API format for function calling

Requirements:
- AWS credentials configured (IAM role or access keys)
- SageMaker endpoint deployed with Arcee-Agent
- MCP servers available locally or remotely

Author: CS-E4660 Course Project
Date: October 2025
"""

import asyncio
import json
import os
from typing import Dict, List, Any, Optional
import boto3
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPSageMakerBridge:
    """
    Bridge between MCP tool servers and AWS SageMaker endpoint.
    
    This class manages:
    1. Connections to multiple MCP servers
    2. Tool discovery and registration
    3. Tool execution via MCP
    4. SageMaker endpoint invocation with OpenAI-compatible format
    """
    
    def __init__(self, endpoint_name: str, region: str = None):
        """
        Initialize the bridge.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            region: AWS region (default: from boto3 session)
        """
        self.endpoint_name = endpoint_name
        self.region = region or boto3.Session().region_name
        self.runtime_client = boto3.client(
            'runtime.sagemaker',
            region_name=self.region
        )
        self.mcp_sessions: Dict[str, ClientSession] = {}
        self.available_tools: List[Dict[str, Any]] = []
        
    async def add_mcp_server(self, name: str, server_script_path: str):
        """
        Connect to an MCP server and register its tools.
        
        Args:
            name: Friendly name for the server
            server_script_path: Path to the MCP server Python script
        """
        print(f"Connecting to MCP server '{name}'...")
        
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        )
        
        # Establish stdio transport
        stdio_transport = await stdio_client(server_params)
        stdio, write = stdio_transport
        
        # Create session
        session = ClientSession(stdio, write)
        await session.initialize()
        
        # Store session
        self.mcp_sessions[name] = session
        
        # Discover tools
        tools_response = await session.list_tools()
        tool_count = 0
        
        for tool in tools_response.tools:
            self.available_tools.append({
                "name": f"{name}_{tool.name}",
                "description": tool.description or f"Tool from {name} server",
                "input_schema": tool.inputSchema,
                "server": name,
                "original_name": tool.name
            })
            tool_count += 1
        
        print(f"✓ Connected to '{name}' with {tool_count} tools")
    
    def format_tools_for_sagemaker(self) -> List[Dict[str, Any]]:
        """
        Format MCP tools for SageMaker's OpenAI-compatible API.
        
        Returns:
            List of tool definitions in OpenAI function format
        """
        formatted_tools = []
        
        for tool in self.available_tools:
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            })
        
        return formatted_tools
    
    async def invoke_tool(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """
        Execute an MCP tool.
        
        Args:
            tool_name: Name of the tool (including server prefix)
            parameters: Tool parameters
            
        Returns:
            Tool execution result as string
        """
        # Find tool information
        tool_info = next(
            (t for t in self.available_tools if t["name"] == tool_name),
            None
        )
        
        if not tool_info:
            return f"Error: Tool '{tool_name}' not found"
        
        server_name = tool_info["server"]
        original_name = tool_info["original_name"]
        session = self.mcp_sessions[server_name]
        
        try:
            # Call MCP tool
            result = await session.call_tool(original_name, parameters)
            
            # Extract text from result
            if result.content and len(result.content) > 0:
                return result.content[0].text
            return str(result)
            
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def invoke_sagemaker_endpoint(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Invoke the SageMaker endpoint with OpenAI-compatible format.
        
        Args:
            messages: Conversation messages
            tools: Available tools for function calling
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response from the endpoint
        """
        payload = {
            "model": "tgi",  # Text Generation Inference
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Body=json.dumps(payload)
            )
            
            result = json.loads(response["Body"].read().decode("utf8"))
            return result
            
        except Exception as e:
            raise RuntimeError(f"SageMaker endpoint error: {str(e)}")


class ArceeAgentSageMakerMCP:
    """
    Complete agent implementation using Arcee Agent on SageMaker with MCP tools.
    
    Features:
    - AWS SageMaker endpoint for scalable inference
    - MCP integration for standardized tool access
    - OpenAI-compatible function calling
    - Conversation history management
    - Error handling and retry logic
    """
    
    def __init__(self, endpoint_name: str, region: str = None):
        """
        Initialize the agent.
        
        Args:
            endpoint_name: SageMaker endpoint name
            region: AWS region (optional)
        """
        self.endpoint_name = endpoint_name
        self.bridge = MCPSageMakerBridge(endpoint_name, region)
        self.conversation_history: List[Dict[str, Any]] = []
        self.system_prompt = (
            "You are a helpful AI assistant with access to various tools. "
            "Use the available tools when needed to answer user questions "
            "accurately and efficiently."
        )
    
    async def setup(self, mcp_servers: Dict[str, str]):
        """
        Setup MCP servers.
        
        Args:
            mcp_servers: Dictionary mapping server names to script paths
        """
        print("\n" + "=" * 60)
        print("Setting up Arcee Agent on AWS SageMaker with MCP")
        print("=" * 60)
        print(f"Endpoint: {self.endpoint_name}")
        print(f"Region: {self.bridge.region}")
        print(f"MCP Servers: {len(mcp_servers)}")
        print()
        
        for name, script_path in mcp_servers.items():
            await self.bridge.add_mcp_server(name, script_path)
        
        print(f"\n✓ Total tools available: {len(self.bridge.available_tools)}")
        print("\nAvailable tools:")
        for tool in self.bridge.available_tools:
            print(f"  • {tool['name']}: {tool['description']}")
        print()
    
    async def query(self, user_message: str) -> str:
        """
        Process a user query with tool calling support.
        
        Args:
            user_message: User's question or request
            
        Returns:
            Agent's response
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        # Get available tools
        tools = self.bridge.format_tools_for_sagemaker()
        
        # Add system prompt if this is the first message
        messages = self.conversation_history.copy()
        if len(messages) == 1:
            messages.insert(0, {
                "role": "system",
                "content": self.system_prompt
            })
        
        # Call SageMaker endpoint
        try:
            response = self.bridge.invoke_sagemaker_endpoint(
                messages=messages,
                tools=tools,
                max_tokens=2048
            )
        except Exception as e:
            error_msg = f"Error calling SageMaker endpoint: {str(e)}"
            print(f"\n⚠ {error_msg}")
            return error_msg
        
        # Extract response message
        message = response["choices"][0]["message"]
        
        # Check if tool calls are needed
        if "tool_calls" in message and message["tool_calls"]:
            # Execute tool calls
            for tool_call in message["tool_calls"]:
                function = tool_call["function"]
                tool_name = function["name"]
                tool_args = json.loads(function["arguments"])
                
                print(f"\n🔧 Calling tool: {tool_name}")
                print(f"   Parameters: {tool_args}")
                
                # Execute via MCP
                tool_result = await self.bridge.invoke_tool(
                    tool_name,
                    tool_args
                )
                
                print(f"   Result: {tool_result[:100]}...")
                
                # Add tool result to history
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result
                })
            
            # Get final response with tool results
            try:
                final_response = self.bridge.invoke_sagemaker_endpoint(
                    messages=self.conversation_history,
                    tools=tools,
                    max_tokens=2048
                )
                
                final_message = final_response["choices"][0]["message"]
                final_content = final_message["content"]
            except Exception as e:
                final_content = f"Error generating final response: {str(e)}"
        else:
            # No tools needed
            final_content = message.get("content", "")
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": final_content
        })
        
        return final_content
    
    async def chat_loop(self):
        """Interactive chat interface."""
        print("\n" + "=" * 60)
        print("Arcee Agent on AWS SageMaker with MCP - Interactive Chat")
        print("=" * 60)
        print("Type your questions below. Type 'quit' or 'exit' to end.")
        print("=" * 60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print()  # Add blank line
                response = await self.query(user_input)
                print(f"\nAgent: {response}\n")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("✓ Conversation history cleared")


async def main():
    """Main entry point."""
    
    # Check for AWS credentials
    try:
        boto3.Session().get_credentials()
    except Exception:
        print("❌ AWS credentials not found!")
        print("\nPlease configure AWS credentials using one of:")
        print("  1. AWS IAM role (recommended for EC2/Lambda)")
        print("  2. Environment variables:")
        print("     export AWS_ACCESS_KEY_ID=your_key")
        print("     export AWS_SECRET_ACCESS_KEY=your_secret")
        print("     export AWS_DEFAULT_REGION=us-east-1")
        print("  3. AWS CLI: aws configure")
        return
    
    # Configuration
    endpoint_name = os.getenv(
        "SAGEMAKER_ENDPOINT",
        "Arcee-Agent-2025-01-13-10-30-00"  # Replace with your endpoint
    )
    
    print("\n⚠️  IMPORTANT: Make sure you have:")
    print("  1. Activated your virtual environment")
    print("     Run: source venv/bin/activate")
    print("  2. Deployed Arcee-Agent to SageMaker")
    print(f"  3. Configured endpoint name: {endpoint_name}")
    print()
    
    # Create agent
    agent = ArceeAgentSageMakerMCP(endpoint_name)
    
    # Setup MCP servers
    try:
        await agent.setup({
            "weather": "src/servers/weather_server.py",
            "calculator": "src/servers/calculator_server.py"
        })
    except Exception as e:
        print(f"❌ Error setting up MCP servers: {e}")
        print("\nMake sure MCP server scripts exist in src/servers/")
        return
    
    # Start chat
    await agent.chat_loop()


if __name__ == "__main__":
    # Always remind about virtual environment
    print("\n" + "=" * 70)
    print("Remember to activate virtual environment before running:")
    print("  source venv/bin/activate  (Linux/macOS)")
    print("  venv\\Scripts\\activate     (Windows)")
    print("=" * 70)
    
    asyncio.run(main())
