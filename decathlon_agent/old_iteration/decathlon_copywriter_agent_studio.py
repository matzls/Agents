from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from typing_extensions import TypedDict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Schema definitions
class CopyComponent(BaseModel):
    component_type: str = Field(description="Type of the copy component")
    element_type: str = Field(description="Type of the element")
    char_limit: int = Field(description="Maximum character limit")
    token_limit: int = Field(description="Maximum token limit")
    audience: str = Field(description="Target audience")

class InputSchema(BaseModel):
    components: List[CopyComponent] = Field(default_factory=list)
    max_attempts: int = Field(default=3)

class State(BaseModel):
    input: InputSchema
    generated_content: str = Field(default="")
    validation_results: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    attempt_count: int = Field(default=0)
    status: str = Field(default="initialized")
    generation_history: List[Dict[str, Any]] = Field(default_factory=list)

# Reuse your existing functions with modifications for Studio compatibility
@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)

def generate_content(state: State) -> State:
    """Generate copy using LLM."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            max_retries=3,
            request_timeout=30
        )
        
        # Process first component (for demo purposes)
        component = state.input.components[0]
        
        prompt = f"""Du bist ein Decathlon CRM Copywriter. 
Erstelle motivierenden Content für {component.audience}-Enthusiasten.

WICHTIGE REGELN:
- Keine direkte Anrede
- Keine Click To Actions oder Verlinkungen
- Exakte Zeichenlänge: {component.char_limit} Zeichen
- Token Limit: {component.token_limit} Tokens
- Der Content muss spezifisch auf {component.audience} ausgerichtet sein

Erstelle den Text in deutscher Sprache."""

        response = generate_with_retry(llm, prompt)
        
        # Update state
        state.generated_content = response.content
        state.attempt_count += 1
        state.status = "generated"
        state.generation_history.append({
            "attempt_number": state.attempt_count,
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state

    except Exception as e:
        state.errors.append(f"Generation error: {str(e)}")
        state.status = "error"
        return state

def validate_content(state: State) -> State:
    """Validate the generated content."""
    try:
        component = state.input.components[0]
        content = state.generated_content.strip()
        
        # Basic validation
        validation_results = {
            "char_count": len(content),
            "within_char_limit": len(content) <= component.char_limit,
            "is_empty": not content
        }
        
        state.validation_results = validation_results
        state.status = "completed" if validation_results["within_char_limit"] else "validation_failed"
        
        return state
        
    except Exception as e:
        state.errors.append(f"Validation error: {str(e)}")
        state.status = "error"
        return state

def should_continue(state: State) -> str:
    """Determine if generation should continue."""
    if state.status == "completed":
        return END
    if state.attempt_count >= state.input.max_attempts:
        return END
    if state.status == "validation_failed":
        return "generate"
    return END

# Create the workflow graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate", generate_content)
workflow.add_node("validate", validate_content)

# Add edges
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")

# Add conditional edges
workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        END: END,
        "generate": "generate"
    }
)

# Compile the graph
graph = workflow.compile()

# Example input for testing in LangGraph Studio
example_input = {
    "input": {
        "components": [
            {
                "component_type": "headline",
                "element_type": "title",
                "char_limit": 50,
                "token_limit": 20,
                "audience": "Schwimmen"
            }
        ],
        "max_attempts": 3
    }
} 