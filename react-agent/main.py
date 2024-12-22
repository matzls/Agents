import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import asyncio
from react_agent import graph
from react_agent.state import State, InputState
from react_agent.configuration import Configuration
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

async def main():
    # Initialize the state with a test message
    initial_state = State(
        messages=[HumanMessage(content="What is the weather in San Francisco?")]
    )
    
    # Create configuration
    config = RunnableConfig(
        configurable={
            "model": "ollama/llama3.2:3b",  # Adjust model name as needed
            "max_search_results": 5
        }
    )
    
    # Run the graph with initial state and config
    result = await graph.ainvoke(initial_state, config)
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 
