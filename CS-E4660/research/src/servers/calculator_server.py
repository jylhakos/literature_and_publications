"""
Calculator MCP Server: Provides mathematical calculation tools via MCP protocol
"""

import asyncio
import math
from mcp.server import Server
from mcp.types import Tool, TextContent


# Create MCP server
app = Server("calculator-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="calculate_arithmetic",
            description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The arithmetic operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        ),
        Tool(
            name="calculate_advanced",
            description="Perform advanced mathematical operations (power, sqrt, log)",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["power", "sqrt", "log", "sin", "cos", "tan"],
                        "description": "The advanced operation to perform"
                    },
                    "value": {
                        "type": "number",
                        "description": "The input value"
                    },
                    "exponent": {
                        "type": "number",
                        "description": "Exponent for power operation (optional)",
                        "default": 2
                    }
                },
                "required": ["operation", "value"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool call"""
    
    try:
        if name == "calculate_arithmetic":
            operation = arguments["operation"]
            a = float(arguments["a"])
            b = float(arguments["b"])
            
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return [TextContent(
                        type="text",
                        text="Error: Division by zero"
                    )]
                result = a / b
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown operation: {operation}"
                )]
            
            return [TextContent(
                type="text",
                text=f"Result: {a} {operation} {b} = {result}"
            )]
        
        elif name == "calculate_advanced":
            operation = arguments["operation"]
            value = float(arguments["value"])
            
            if operation == "power":
                exponent = float(arguments.get("exponent", 2))
                result = math.pow(value, exponent)
                return [TextContent(
                    type="text",
                    text=f"Result: {value}^{exponent} = {result}"
                )]
            
            elif operation == "sqrt":
                if value < 0:
                    return [TextContent(
                        type="text",
                        text="Error: Cannot calculate square root of negative number"
                    )]
                result = math.sqrt(value)
                return [TextContent(
                    type="text",
                    text=f"Result: √{value} = {result}"
                )]
            
            elif operation == "log":
                if value <= 0:
                    return [TextContent(
                        type="text",
                        text="Error: Logarithm undefined for non-positive numbers"
                    )]
                result = math.log(value)
                return [TextContent(
                    type="text",
                    text=f"Result: ln({value}) = {result}"
                )]
            
            elif operation in ["sin", "cos", "tan"]:
                result = getattr(math, operation)(value)
                return [TextContent(
                    type="text",
                    text=f"Result: {operation}({value}) = {result}"
                )]
            
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown operation: {operation}"
                )]
        
        else:
            return [TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]
    
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"Error: {str(e)}"
        )]


if __name__ == "__main__":
    import sys
    print("Starting Calculator MCP Server...", file=sys.stderr)
    asyncio.run(app.run())
