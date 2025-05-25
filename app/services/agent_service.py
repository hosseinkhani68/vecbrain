from langchain_openai import ChatOpenAI
from langchain.agents import Tool
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from app.config import get_settings
from app.services.document_service import DocumentService
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import requests
import json
from datetime import datetime
import math

settings = get_settings()

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

class CalculatorTool(BaseTool):
    name: str = "calculator"
    description: str = "Perform mathematical calculations"

    def _run(self, expression: str) -> str:
        """Run the calculator tool."""
        try:
            # Basic safety check
            if any(keyword in expression.lower() for keyword in ['import', 'eval', 'exec', 'os', 'sys']):
                return "Error: Invalid expression"
            
            # Use eval for calculation (with safety measures)
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error calculating: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Run the calculator tool asynchronously."""
        return self._run(expression)

class AgentService:
    def __init__(self, document_service: DocumentService):
        self.document_service = document_service
        self.llm = ChatOpenAI(
            model="gpt-4-1106-preview",
            temperature=0.7,
            openai_api_key=settings.openai_api_key
        )
        self.tools = self._create_tools()
        self.agent = self._create_agent()

    def _create_tools(self) -> List[BaseTool]:
        """Create the tools for the agent."""
        return [
            SearchTool(document_service=self.document_service),
            CalculatorTool(),
            Tool(
                name="current_time",
                func=lambda _: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                description="Useful for getting the current time"
            )
        ]

    def _create_agent(self) -> AgentExecutor:
        """Create the agent with the tools."""
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            system_prompt="""You are a helpful AI assistant that can search through documents and perform calculations.
            When searching documents, provide clear and concise summaries of the relevant information found.
            When performing calculations, show your work and explain the steps taken."""
        )
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    async def run_agent(self, query: str) -> str:
        """Run the agent with a query."""
        try:
            response = await self.agent.ainvoke({"input": query})
            return response["output"]
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