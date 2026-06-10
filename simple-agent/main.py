from mcp import ClientSession, StdioServerParameters 
from mcp.client.stdio import stdio_client 
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv 
import asyncio
import os 

load_dotenv()

model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature= 0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# connect to mcp tool server
server_params = StdioServerParameters(
    command="npx",
    env={
        "FIRECRAWL_API_KEY": os.getenv("FIRECRAWL_API_KEY")
    },
    args=["firecrawl-mcp"] # bg process to run mcp client to connect to server
)

async def main():
    # connecting to client 
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            # give agent access to model and mcp toosl
            agent = create_react_agent(model, tools)

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can scrape websites, crawl pages, and extract data. using Firecrawl tools. Think step by step and use the appropriate tools to help the user."
                }
            ]
            # unpack the tool names
            print("Available Tools - ", *[tool.name for tool in tools])
            print("-" * 60)

            while True:
                user_input = input("\nYou: ")
                if user_input == "quit":
                    print("Goodbye")
                    break 
                
                messages.append({"role": "user", "content": user_input[:175000]})

                try:
                    # async invoke agent with message state 
                    agent_response = await agent.ainvoke({"messages": messages})

                    ai_message = agent_response["messages"][-1].content
                    print("\nAgent:", ai_message)
                except Exception as e:
                    print("Error: ", e)

if __name__ == "__main__":
    asyncio.run(main())
