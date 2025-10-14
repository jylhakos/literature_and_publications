"""
MCP to LangChain Bridge: Converts MCP tools to LangChain-compatible tools
"""

from langchain.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import asyncio


class MCPToolWrapper(BaseTool):
    """Wrapper to expose MCP tools as LangChain tools"""
    
    name: str
    description: str
    mcp_session: Any = Field(exclude=True)
    tool_schema: Dict = Field(exclude=True)
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(self, **kwargs) -> str:
        """Synchronous execution wrapper"""
        return asyncio.run(self._arun(**kwargs))
    
    async def _arun(self, **kwargs) -> str:
        """Execute MCP tool call asynchronously"""
        try:
            result = await self.mcp_session.call_tool(
                self.name, kwargs
            )
            
            # Extract text content from result
            if isinstance(result.content, list):
                text_parts = []
                for content_item in result.content:
                    if hasattr(content_item, 'text'):
                        text_parts.append(content_item.text)
                return "\n".join(text_parts)
            
            return str(result.content)
            
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"


class MCPBridge:
    """Bridge between MCP servers and LangChain framework"""
    
    def __init__(self):
        """Initialize MCP Bridge"""
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: List[MCPToolWrapper] = []
    
    async def connect_to_server(self, name: str, server_path: str):
        """
        Connect to an MCP server
        
        Args:
            name: Identifier for this server
            server_path: Path to server script
        """
        is_python = server_path.endswith('.py')
        command = "python" if is_python else "node"
        
        server_params = StdioServerParameters(
            command=command,
            args=[server_path],
            env=None
        )
        
        # Create MCP client
        stdio, write = await stdio_client(server_params)
        session = ClientSession(stdio, write)
        await session.initialize()
        
        self.sessions[name] = session
        
        # Convert MCP tools to LangChain tools
        response = await session.list_tools()
        for tool in response.tools:
            langchain_tool = MCPToolWrapper(
                name=tool.name,
                description=tool.description,
                mcp_session=session,
                tool_schema=tool.inputSchema
            )
            self.tools.append(langchain_tool)
        
        print(f"Connected to MCP server '{name}': {len(response.tools)} tools available")
    
    async def connect_multiple(self, servers: Dict[str, str]):
        """
        Connect to multiple MCP servers
        
        Args:
            servers: Dictionary mapping server names to paths
        """
        for name, path in servers.items():
            try:
                await self.connect_to_server(name, path)
            except Exception as e:
                print(f"Warning: Failed to connect to {name}: {e}")
        
        print(f"\nTotal tools available: {len(self.tools)}")
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all LangChain-compatible tools
        
        Returns:
            List of LangChain BaseTool instances
        """
        return self.tools
    
    async def cleanup(self):
        """Clean up all MCP sessions"""
        for name, session in self.sessions.items():
            try:
                # Close session if it has a close method
                if hasattr(session, 'close'):
                    await session.close()
            except Exception as e:
                print(f"Warning: Error closing session {name}: {e}")
