from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
import requests
import json
from datetime import datetime
import math
from app.config import get_settings
from app.services.document_service import DocumentService

# Custom tool for calculations
class CalculatorInput(BaseModel):
    expression: str = Field(description="The mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Useful for performing mathematical calculations"
    args_schema: Type[BaseModel] = CalculatorInput

    def _run(self, expression: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            # Safely evaluate the expression
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error calculating: {str(e)}"

# Custom tool for weather information
class WeatherInput(BaseModel):
    location: str = Field(description="The city name to get weather for")

class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Useful for getting weather information for a location"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, location: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            # This is a mock implementation. In a real app, you'd use a weather API
            return f"Weather in {location}: Sunny, 25°C"
        except Exception as e:
            return f"Error getting weather: {str(e)}"

# Custom tool for document search
class SearchInput(BaseModel):
    query: str = Field(description="The search query")

class SearchTool(BaseTool):
    name: str = "search_documents"
    description: str = "Search for relevant information in the document store"
    document_service: DocumentService = Field(description="Document service instance for searching documents")

    def _run(self, query: str) -> str:
        """Run the search tool."""
        try:
            results = self.document_service.search_documents(query)
            if not results:
                return "No relevant documents found."
            
            response = "Here are the relevant documents:\n\n"
            for i, result in enumerate(results, 1):
                response += f"{i}. {result['text']}\n"
                if result.get('metadata'):
                    response += f"   Source: {result['metadata'].get('source', 'Unknown')}\n"
                response += "\n"
            return response
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Run the search tool asynchronously."""
        return self._run(query)

class AgentService:
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.7,
            openai_api_key=get_settings().openai_api_key
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self):
        """Create a list of tools for the agent to use."""
        return [
            CalculatorTool(),
            WeatherTool(),
            SearchTool(document_service=self.document_service),
            Tool(
                name="current_time",
                func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                description="Useful for getting the current time"
            ),
            Tool(
                name="format_number",
                func=lambda x: f"{float(x):,.2f}",
                description="Useful for formatting numbers with commas and decimals"
            )
        ]

    def _create_agent(self):
        """Create an agent with the specified tools."""
        agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        return agent

    async def run_agent(self, query: str) -> str:
        """Run the agent with a query."""
        try:
            response = await self.agent.arun(query)
            return response
        except Exception as e:
            return f"Error running agent: {str(e)}"

    async def process_complex_query(self, query: str) -> dict:
        """Process a complex query that might require multiple tools."""
        try:
            # Run the agent
            response = await self.run_agent(query)
            
            # Get the tools used
            tools_used = self.agent.tools_used if hasattr(self.agent, 'tools_used') else []
            
            return {
                "response": response,
                "tools_used": tools_used,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            } 