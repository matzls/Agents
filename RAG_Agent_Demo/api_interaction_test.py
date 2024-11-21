from langgraph_sdk import get_client
import asyncio
from rich import print as rprint
from rich.console import Console

console = Console()

async def main():
    client = get_client(url="http://localhost:49514")
    
    # Get the last assistant
    assistants = await client.assistants.search()
    last_assistant = assistants[-1]
    
    # Print the last assistant
    rprint("[bold green]Last Assistant:[/bold green]")
    console.print(last_assistant, style="bold blue")

    # Get the graph info using the correct method
    try:
        graph_info = await client.assistants.get_graph(last_assistant['assistant_id'])
        
        # Print graph info
        rprint("[bold blue]Graph Info:[/bold blue]")
        console.print(graph_info, style="bold green")
    
    except Exception as e:
        rprint(f"[bold red]Error retrieving graph info: {e}[/bold red]")


    try:
        threads = await client.threads.search()
        rprint("[bold blue]Threads:[/bold blue]")
        console.print(threads, style="bold green")
    except Exception as e:
        rprint(f"[bold red]Error retrieving threads: {e}[/bold red]")   


async def thread_test():
    client = get_client(url="http://localhost:49514")
    threads = await client.threads.search()
    rprint("[bold blue]Threads:[/bold blue]")
    console.print(threads, style="bold green")  





asyncio.run(thread_test())
