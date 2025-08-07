from langchain_groq import ChatGroq   # groq LLM
from langchain_community.tools.tavily_search import TavilySearchResults   # Tavily LLM

# AI Agent
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage   # separate answer AI or human

# config
from app.config.settings import settings

def get_response_from_ai_agents(llm_id, query, allow_search, system_prompt):

    llm = ChatGroq(model=llm_id)

    # allow search or not / allow only top two results
    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    state = {"messages" : query}

    response = agent.invoke(state)

    messages = response.get("messages")

    ai_messages = [message.content for message in messages if isinstance(message,AIMessage)]

    # latest reply
    return ai_messages[-1]