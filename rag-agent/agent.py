import datetime
from typing import Optional
from sqlalchemy.orm import Session

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
except ImportError:
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.runnables.history import RunnableWithMessageHistory

from config import settings
from exceptions import LLMProviderError
from prompts import AGENT_SYSTEM_PROMPT
from tools import search_documents, web_search, get_current_time
from memory import MySQLChatMessageHistory

def get_agent_executor(db_session: Session, model: Optional[str] = None) -> RunnableWithMessageHistory:
    """
    Constructs and returns a RunnableWithMessageHistory wrapper around the LangChain AgentExecutor.
    
    This agent uses Groq to run tasks, search documents, and store history in MySQL.
    """
    # 1. Initialize the LLM based on Groq provider
    if not settings.GROQ_API_KEY or "your_groq_api" in settings.GROQ_API_KEY.lower():
        raise LLMProviderError(
            message="Groq API Key is missing or invalid. Please configure GROQ_API_KEY in your .env file.",
            error_code="GROQ_API_KEY_MISSING"
        )
    from langchain_groq import ChatGroq
    
    # Pick the requested model, or fallback to config
    selected_model = model or settings.LLM_MODEL
    # Fallback to default Groq model if selected model is empty or references an old OpenAI model
    if not selected_model or any(openai_indicator in selected_model.lower() for openai_indicator in ["gpt", "openai"]):
        selected_model = "llama-3.1-8b-instant"
        
    llm = ChatGroq(
        model=selected_model,
        temperature=settings.LLM_TEMPERATURE,
        groq_api_key=settings.GROQ_API_KEY
    )

    # Patch ChatGroq streaming to prevent XML-to-JSON parsing issues on Groq server side
    # under streaming tool calling execution.
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.messages import AIMessageChunk

    def patch_groq_streaming(llm_instance):
        def custom_stream(self, *args, **kwargs):
            # Call non-streaming generate method to prevent API stream validation errors
            result = self._generate(*args, **kwargs)
            if result.generations:
                gen = result.generations[0]
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=gen.message.content,
                        additional_kwargs=gen.message.additional_kwargs,
                        tool_calls=gen.message.tool_calls,
                        response_metadata=gen.message.response_metadata,
                        id=gen.message.id
                    ),
                    generation_info=gen.generation_info
                )
        object.__setattr__(llm_instance, "_stream", custom_stream.__get__(llm_instance, ChatGroq))

    patch_groq_streaming(llm)




    # 2. Define the tools
    tools = [search_documents, web_search, get_current_time]

    # 3. Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", AGENT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 4. Build the tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)

    # 5. Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,   # Reduced from 10 to save tokens
        handle_parsing_errors=True
    )

    # 6. Wrap with MySQL message history logic.
    # history_trimmer keeps only the last 10 messages per request to prevent
    # context_length_exceeded errors during long chat sessions.
    def trimmed_history_factory(session_id: str):
        history = MySQLChatMessageHistory(session_id, db_session)
        # Trim to last 10 messages (5 turns) before passing to agent
        raw_messages = history.messages
        if len(raw_messages) > 10:
            history._messages_override = raw_messages[-10:]
        return history

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history=lambda session_id: MySQLChatMessageHistory(session_id, db_session),
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )

    return agent_with_history
