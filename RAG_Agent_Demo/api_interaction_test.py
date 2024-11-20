from langgraph_sdk import get_client
import os
import asyncio

# get my api key from the environment
api_key = os.getenv("LANGCHAIN_API_KEY")

async def main():
    client = get_client(url="http://localhost:57515")
    # get default assistant
    assistants = await client.assistants.search()
    assistant = assistants[0]
    print(assistant)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())


    