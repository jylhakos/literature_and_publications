"""
Claude MCP Agent - Approach A
AI Agent using Claude SDK with Model Context Protocol integration
"""

import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()


class MCPClaudeAgent:
    """AI Agent using Claude with MCP server integration"""
    
    def __init__(self):
        """Initialize the MCP Claude Agent"""
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        self.available_tools = []
    
    async def connect_to_server(self, server_script_path: str):
        """
        Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # Establish connection
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        
        # Create session
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        
        # Initialize and discover tools
        await self.session.initialize()
        response = await self.session.list_tools()
        self.available_tools = response.tools
        
        print(f"Connected to server with tools: "
              f"{[tool.name for tool in self.available_tools]}")
    
    async def process_query(self, query: str) -> str:
        """
        Process query using Claude with MCP tools
        
        Args:
            query: User's question or request
            
        Returns:
            Agent's response as string
        """
        messages = [{"role": "user", "content": query}]
        
        # Format tools for Claude API
        tools_for_api = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in self.available_tools]
        
        # Initial Claude API call
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            messages=messages,
            tools=tools_for_api
        )
        
        # Agentic loop: handle tool calls
        final_text = []
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            assistant_content = []
            has_tool_use = False
            
            for content in response.content:
                if content.type == 'text':
                    final_text.append(content.text)
                    assistant_content.append(content)
                    
                elif content.type == 'tool_use':
                    has_tool_use = True
                    tool_name = content.name
                    tool_args = content.input
                    
                    # Execute tool via MCP
                    print(f"[Calling {tool_name} with {tool_args}]")
                    result = await self.session.call_tool(
                        tool_name, tool_args
                    )
                    
                    assistant_content.append(content)
                    
                    # Add assistant and tool result to messages
                    messages.append({
                        "role": "assistant",
                        "content": assistant_content
                    })
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }]
                    })
                    
                    # Get next response from Claude
                    response = self.anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=4000,
                        messages=messages,
                        tools=tools_for_api
                    )
                    break
            
            if not has_tool_use:
                # No more tool calls, we're done
                break
        
        return "\n".join(final_text)
    
    async def chat_loop(self):
        """Run interactive chat loop"""
        print("\n" + "="*60)
        print("MCP Claude Agent Started!")
        print("="*60)
        print("Type your queries or 'quit' to exit.\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not query:
                    continue
                
                response = await self.process_query(query)
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        print("\nExample:")
        print("  python client.py ../servers/weather_server.py")
        sys.exit(1)
    
    agent = MCPClaudeAgent()
    try:
        await agent.connect_to_server(sys.argv[1])
        await agent.chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
