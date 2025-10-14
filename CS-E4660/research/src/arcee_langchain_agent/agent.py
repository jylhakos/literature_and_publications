"""
Arcee Agent with LangChain and MCP - Approach B
AI Agent using Arcee Agent SLM with LangChain framework and MCP integration
"""

from langchain.llms.base import LLM
from typing import Optional, List, Any, Dict
from pydantic import Field
import asyncio
import re
import xml.etree.ElementTree as ET

from .config import ArceeAgentConfig
from .mcp_bridge import MCPBridge, BaseTool


class ArceeLangChainLLM(LLM):
    """Custom LangChain LLM wrapper for Arcee Agent"""
    
    model: Any = Field(exclude=True)
    tokenizer: Any = Field(exclude=True)
    max_new_tokens: int = 2048
    temperature: float = 0.7
    
    class Config:
        arbitrary_types_allowed = True
    
    @property
    def _llm_type(self) -> str:
        return "arcee-agent"
    
    def _call(self, 
              prompt: str, 
              stop: Optional[List[str]] = None,
              **kwargs) -> str:
        """Execute model inference"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate response
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode output
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def format_tool_prompt(self, 
                          tools: List[BaseTool],
                          query: str) -> str:
        """Format prompt with tools for Arcee Agent"""
        
        tools_xml = "<tools>\n"
        for tool in tools:
            tools_xml += f"  <tool>\n"
            tools_xml += f"    <name>{tool.name}</name>\n"
            tools_xml += f"    <description>{tool.description}</description>\n"
            tools_xml += f"  </tool>\n"
        tools_xml += "</tools>\n\n"
        
        prompt = f"""In this environment, you have access to a set of tools you can use to answer the user's question.

You may call them like this:
<function_calls>
  <invoke>
    <tool_name>$TOOL_NAME</tool_name>
    <parameters>
      <$PARAMETER_NAME>$PARAMETER_VALUE</$PARAMETER_NAME>
      ...
    </parameters>
  </invoke>
</function_calls>

Here are the tools available:
{tools_xml}

User Query: {query}

Your response:"""
        
        return prompt


class ArceeOutputParser:
    """Parse tool calls from Arcee Agent's XML-formatted output"""
    
    @staticmethod
    def extract_tool_calls(output: str) -> List[Dict[str, Any]]:
        """Extract tool calls from XML output"""
        tool_calls = []
        
        # Find all function_calls blocks
        pattern = r'<function_calls>(.*?)</function_calls>'
        matches = re.findall(pattern, output, re.DOTALL)
        
        for match in matches:
            try:
                # Parse XML
                root = ET.fromstring(f"<root>{match}</root>")
                
                for invoke in root.findall('invoke'):
                    tool_name_elem = invoke.find('tool_name')
                    if tool_name_elem is None:
                        continue
                    
                    tool_name = tool_name_elem.text
                    params_elem = invoke.find('parameters')
                    
                    # Extract parameters
                    parameters = {}
                    if params_elem is not None:
                        for param in params_elem:
                            if param.text:
                                parameters[param.tag] = param.text
                    
                    tool_calls.append({
                        'tool_name': tool_name,
                        'parameters': parameters
                    })
            
            except Exception as e:
                print(f"Error parsing tool call: {e}")
                continue
        
        return tool_calls
    
    @staticmethod
    def remove_tool_calls(output: str) -> str:
        """Remove tool call XML from output"""
        pattern = r'<function_calls>.*?</function_calls>'
        cleaned = re.sub(pattern, '', output, flags=re.DOTALL)
        return cleaned.strip()


class ArceeAgentWithMCP:
    """Arcee Agent integrated with MCP via LangChain"""
    
    def __init__(self, use_quantization: bool = False):
        """
        Initialize Arcee Agent with MCP
        
        Args:
            use_quantization: Whether to use 4-bit quantization (requires GPU)
        """
        print("Initializing Arcee Agent with MCP...")
        
        # Load Arcee model
        if use_quantization:
            model, tokenizer = ArceeAgentConfig.load_model_quantized()
        else:
            model, tokenizer = ArceeAgentConfig.load_model()
        
        # Create LangChain LLM wrapper
        self.llm = ArceeLangChainLLM(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.7
        )
        
        # Initialize MCP bridge
        self.mcp_bridge = MCPBridge()
        self.parser = ArceeOutputParser()
        
        print("Arcee Agent initialized successfully")
    
    async def setup(self, mcp_servers: Dict[str, str]):
        """
        Setup MCP connections
        
        Args:
            mcp_servers: Dictionary mapping server names to paths
        """
        print("\nConnecting to MCP servers...")
        await self.mcp_bridge.connect_multiple(mcp_servers)
    
    async def execute_with_tools(self, query: str) -> str:
        """
        Execute query with explicit tool calling
        
        Args:
            query: User's question or request
            
        Returns:
            Agent's response
        """
        # Format prompt with tools
        prompt = self.llm.format_tool_prompt(
            self.mcp_bridge.get_tools(),
            query
        )
        
        # Generate response
        output = self.llm._call(prompt)
        
        # Parse tool calls
        tool_calls = self.parser.extract_tool_calls(output)
        
        if not tool_calls:
            # No tools needed, return response
            return self.parser.remove_tool_calls(output)
        
        # Execute tool calls
        print(f"\nExecuting {len(tool_calls)} tool call(s)...")
        tool_results = []
        for call in tool_calls:
            tool_name = call['tool_name']
            parameters = call['parameters']
            
            print(f"  - Calling {tool_name} with {parameters}")
            
            # Find tool in MCP bridge
            tool = next(
                (t for t in self.mcp_bridge.get_tools() 
                 if t.name == tool_name),
                None
            )
            
            if tool:
                try:
                    result = await tool._arun(**parameters)
                    tool_results.append({
                        'tool': tool_name,
                        'result': result
                    })
                except Exception as e:
                    tool_results.append({
                        'tool': tool_name,
                        'result': f"Error: {str(e)}"
                    })
            else:
                tool_results.append({
                    'tool': tool_name,
                    'result': f"Error: Tool {tool_name} not found"
                })
        
        # Format results for final response
        results_text = "\n".join([
            f"Tool {r['tool']} returned: {r['result']}"
            for r in tool_results
        ])
        
        # Generate final response with tool results
        final_prompt = f"""{prompt}

Tool Results:
{results_text}

Based on these results, provide your final answer:"""
        
        final_response = self.llm._call(final_prompt)
        
        return self.parser.remove_tool_calls(final_response)
    
    async def chat_loop(self):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("Arcee Agent with MCP Started!")
        print("="*60)
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                response = await self.execute_with_tools(user_input)
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.mcp_bridge.cleanup()


async def main():
    """Main entry point"""
    import sys
    
    # Define MCP servers
    if len(sys.argv) < 2:
        print("Usage: python agent.py <server1_path> [server2_path] ...")
        print("\nExample:")
        print("  python agent.py ../servers/weather_server.py")
        sys.exit(1)
    
    servers = {}
    for i, server_path in enumerate(sys.argv[1:], 1):
        servers[f"server_{i}"] = server_path
    
    # Create and setup agent
    agent = ArceeAgentWithMCP(use_quantization=False)
    
    try:
        await agent.setup(servers)
        await agent.chat_loop()
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
