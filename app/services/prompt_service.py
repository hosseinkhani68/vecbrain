from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    PromptTemplate
)
from langchain_openai import ChatOpenAI
from typing import List, Dict, Any, Optional
from datetime import datetime

class PromptService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name="gpt-4-1106-preview",
            temperature=0.7,
            max_tokens=1000
        )
        self.templates = self._create_templates()

    def _create_templates(self) -> Dict[str, ChatPromptTemplate]:
        """Create a dictionary of prompt templates for different use cases."""
        return {
            # Chat template with context
            "chat": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are an AI assistant with a deep understanding of various topics.
                    You maintain context from previous conversations and provide detailed, accurate responses.
                    
                    Current conversation history:
                    {history}
                    
                    Additional context:
                    {context}"""
                ),
                HumanMessagePromptTemplate.from_template("{input}")
            ]),

            # Document summarization template
            "summarize": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are an expert at summarizing documents.
                    Create a concise summary that captures the main points and key details.
                    
                    Document to summarize:
                    {document}
                    
                    Requirements:
                    - Keep the summary under {max_length} words
                    - Focus on key points and main ideas
                    - Maintain the original meaning
                    - Use clear, concise language"""
                ),
                HumanMessagePromptTemplate.from_template("Please provide a summary of the document.")
            ]),

            # Question answering template
            "qa": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are an expert at answering questions based on provided context.
                    Use the following context to answer the question.
                    If you cannot answer based on the context, say so.
                    
                    Context:
                    {context}
                    
                    Question:
                    {question}"""
                ),
                HumanMessagePromptTemplate.from_template("Please provide a detailed answer.")
            ]),

            # Code explanation template
            "code_explain": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are an expert programmer who explains code clearly.
                    Explain the following code in detail:
                    
                    Code:
                    {code}
                    
                    Requirements:
                    - Explain the purpose and functionality
                    - Break down complex parts
                    - Highlight important patterns or techniques
                    - Suggest potential improvements"""
                ),
                HumanMessagePromptTemplate.from_template("Please explain this code.")
            ]),

            # Creative writing template
            "creative": ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    """You are a creative writer with a unique style.
                    Create content based on the following prompt:
                    
                    Style: {style}
                    Tone: {tone}
                    Length: {length} words
                    
                    Prompt:
                    {prompt}"""
                ),
                HumanMessagePromptTemplate.from_template("Please create the content.")
            ])
        }

    async def generate_response(
        self,
        template_name: str,
        input_data: Dict[str, Any],
        context: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate a response using the specified template."""
        try:
            # Get the template
            template = self.templates.get(template_name)
            if not template:
                raise ValueError(f"Template '{template_name}' not found")

            # Prepare the input
            template_input = input_data.copy()
            
            # Add context and history if provided
            if context:
                template_input["context"] = context
            if history:
                template_input["history"] = self._format_history(history)

            # Generate the response
            messages = template.format_messages(**template_input)
            response = await self.llm.ainvoke(messages)

            return {
                "response": response.content,
                "template_used": template_name,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history for the prompt."""
        formatted = []
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        return "\n".join(formatted)

    def get_available_templates(self) -> List[str]:
        """Get a list of available template names."""
        return list(self.templates.keys())

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a specific template."""
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return {
            "name": template_name,
            "description": template.messages[0].prompt.template,
            "input_variables": template.input_variables
        } 