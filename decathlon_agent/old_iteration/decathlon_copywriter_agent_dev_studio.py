from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CopyComponent(BaseModel):
    name: str
    char_limit: int
    briefing: str
    audience: str

class InputSchema(BaseModel):
    components: List[CopyComponent] = Field(default_factory=list)
    max_attempts: int = Field(default=3)

class State(BaseModel):
    input: InputSchema
    generated_content: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    attempt_count: int = Field(default=0)
    status: str = Field(default="")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)

def generate_content(state: State) -> State:
    """Generate copy using LLM based on component requirements."""
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.4,
        max_retries=3,
        request_timeout=30
    )

    for component in state.input.components:
        state.attempt_count = 0
        while state.attempt_count < state.input.max_attempts:
            state.attempt_count += 1
            logger.info(f"Starting generation attempt #{state.attempt_count} for component '{component.name}'")
            
            try:
                prompt = f"""Du bist ein Decathlon CRM Copywriter. 
Erstelle motivierenden Content für {component.audience}-Enthusiasten.

KONTEXT UND MARKENIDENTITÄT:
- Freundlich und Einladend: Persönliche, einladende Note
- Begeistert und Positiv: Freude am Sport vermitteln
- Kundenorientiert und Unterstützend: Inspirierende Inhalte
- Einfach und Direkt: Verständliche Sprache
- Spielerisch und Energetisch: Leichter Humor und Dynamik

Briefing: {component.briefing}

WICHTIGE REGELN:
- Keine direkte Anrede
- Keine CTAs oder Verlinkungen im Einführungstext
- Exakte Zeichenlänge: {component.char_limit} Zeichen
- Der Content muss spezifisch auf {component.audience} ausgerichtet sein

Erstelle den Text in deutscher Sprache.
WICHTIG: Der Text MUSS EXAKT {component.char_limit} Zeichen lang sein."""
                
                response = generate_with_retry(llm, prompt)
                generated_content = response.content.strip()
                
                # Validate the content
                char_count = len(generated_content)
                if char_count != component.char_limit:
                    feedback = f"Der Text ist {char_count} Zeichen lang, sollte aber exakt {component.char_limit} Zeichen sein."
                    state.errors.append(feedback)
                    continue  # Retry
                else:
                    # Save the valid content
                    state.generated_content.append({
                        "component": component.name,
                        "content": generated_content
                    })
                    break  # Exit the retry loop

            except Exception as e:
                error_msg = f"Content generation error for component '{component.name}': {str(e)}"
                logger.error(error_msg)
                state.errors.append(error_msg)
                continue  # Retry on error

        if state.attempt_count >= state.input.max_attempts:
            state.errors.append(f"Max attempts reached for component '{component.name}'")

    state.status = "completed" if len(state.generated_content) == len(state.input.components) else "incomplete"
    return state

def validate_content(state: State) -> State:
    """Additional validation if needed."""
    # Implement any additional validation here
    return state

# Graph Construction
builder = StateGraph(State)
builder.add_node("generate", generate_content)
builder.add_node("validate", validate_content)

builder.add_edge(START, "generate")
builder.add_edge("generate", "validate")
builder.add_edge("validate", END)

# Compile
graph = builder.compile()

# Run graph async
async def run_graph(graph, input_data: Dict[str, Any]):
    """Run the graph with the given input data"""
    # Transform the input data to match the expected schema
    if "components" in input_data and isinstance(input_data["components"], dict):
        # Convert dict to list if necessary
        input_data["components"] = [input_data["components"]]
    elif "components" in input_data and isinstance(input_data["components"], list):
        # Ensure each component is properly formatted
        input_data["components"] = [
            comp if isinstance(comp, dict) else comp.dict() 
            for comp in input_data["components"]
        ]
    
    initial_state = State(input=InputSchema(**input_data))
    async with graph.client() as client:
        final_state = await client.run(initial_state)
    return final_state