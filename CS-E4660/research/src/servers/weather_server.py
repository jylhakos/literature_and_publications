"""
Weather MCP Server: Provides weather information via MCP protocol
"""

import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx


# Create MCP server
app = Server("weather-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="get_current_weather",
            description="Get current weather information for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location (e.g., 'London', 'New York')"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units",
                        "default": "celsius"
                    }
                },
                "required": ["location"]
            }
        ),
        Tool(
            name="get_weather_forecast",
            description="Get weather forecast for the next 5 days",
            inputSchema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    },
                    "days": {
                        "type": "integer",
                        "description": "Number of days (1-5)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5
                    }
                },
                "required": ["location"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute tool call"""
    
    if name == "get_current_weather":
        location = arguments["location"]
        units = arguments.get("units", "celsius")
        
        # Mock weather data (replace with real API in production)
        # Example: OpenWeatherMap, WeatherAPI, etc.
        weather_data = await get_mock_weather(location, units)
        
        response_text = f"""Current weather in {location}:
Temperature: {weather_data['temperature']}°{units[0].upper()}
Conditions: {weather_data['conditions']}
Humidity: {weather_data['humidity']}%
Wind Speed: {weather_data['wind_speed']} {weather_data['wind_unit']}"""
        
        return [TextContent(
            type="text",
            text=response_text
        )]
    
    elif name == "get_weather_forecast":
        location = arguments["location"]
        days = arguments.get("days", 3)
        
        # Mock forecast data
        forecast_data = await get_mock_forecast(location, days)
        
        forecast_text = f"Weather forecast for {location} ({days} days):\n\n"
        for day in forecast_data:
            forecast_text += f"{day['date']}: {day['temp_high']}°/{day['temp_low']}° - {day['conditions']}\n"
        
        return [TextContent(
            type="text",
            text=forecast_text
        )]
    
    else:
        return [TextContent(
            type="text",
            text=f"Unknown tool: {name}"
        )]


async def get_mock_weather(location: str, units: str) -> dict:
    """
    Get mock weather data
    In production, replace with actual API call
    """
    # Simulate API delay
    await asyncio.sleep(0.1)
    
    # Mock data
    temp_celsius = 18
    temp = temp_celsius if units == "celsius" else (temp_celsius * 9/5) + 32
    
    return {
        'temperature': round(temp, 1),
        'conditions': 'Partly cloudy',
        'humidity': 65,
        'wind_speed': 12 if units == 'celsius' else 7.5,
        'wind_unit': 'km/h' if units == 'celsius' else 'mph'
    }


async def get_mock_forecast(location: str, days: int) -> list:
    """
    Get mock forecast data
    In production, replace with actual API call
    """
    await asyncio.sleep(0.1)
    
    forecast = []
    for i in range(days):
        forecast.append({
            'date': f"Day {i+1}",
            'temp_high': 20 + i,
            'temp_low': 12 + i,
            'conditions': ['Sunny', 'Cloudy', 'Rainy'][i % 3]
        })
    
    return forecast


if __name__ == "__main__":
    import sys
    print("Starting Weather MCP Server...", file=sys.stderr)
    asyncio.run(app.run())
