from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from typing_extensions import TypedDict
from datetime import datetime
import json
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Schema definitions
@dataclass
class CopyComponent:
    component_type: str
    element_type: str
    char_limit: int
    token_limit: int
    audience: str
    max_attempts: int = 3

class State(TypedDict):
    components: List[CopyComponent]
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]

# Knowledge base loading function
def load_knowledge_base(filepath: str) -> dict:
    """Loads the knowledge base from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

# Reuse your existing functions with modifications for Studio compatibility
@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)

def generate_content(state: State) -> State:
    """Generate copy using LLM."""
    knowledge_base = load_knowledge_base("template_kb.json")
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            max_retries=3,
            request_timeout=30
        )

        # Process first component (for demo purposes)
        component = state["components"][0]

        # Get relevant information from the knowledge base using dictionary key access
        kb_info = knowledge_base.get(component["component_type"], {}).get(component["element_type"], {})
        examples_text = ""
        if kb_info:
            examples_text = "\nBEISPIELE ZUR ORIENTIERUNG:\n"
            for ex in kb_info.get("examples", [])[:2]:
                examples_text += f"Beispiel Output ({len(ex.get('expected_output', ''))} Zeichen): {ex.get('expected_output', '')}\n"

        # Add feedback to the prompt if the previous attempt failed validation
        feedback_text = ""
        if state.get("status") == "validation_failed":
            feedback_text = "\n\nHINWEIS: Der vorherige Versuch ist gescheitert, weil die Zeichen- oder Token-Limits überschritten wurden. Bitte halte dich UNBEDINGT an die vorgegebene Länge."

        prompt = f"""Du bist ein Decathlon CRM Copywriter. 
Erstelle motivierenden Content für {component["audience"]}-Enthusiasten.

WICHTIGE REGELN:
- Keine direkte Anrede
- Keine CTAs oder Verlinkungen
- Exakte Zeichenlänge: {kb_info.get('char_limit', component["char_limit"])} Zeichen
- Token Limit: {kb_info.get('token_limit', component["token_limit"])} Tokens
- Der Content muss spezifisch auf {component["audience"]} ausgerichtet sein

{examples_text}

{feedback_text}

Erstelle den Text in deutscher Sprache."""

        response = generate_with_retry(llm, prompt)

        # Correctly increment the attempt counter
        current_attempt_count = state.get("attempt_count", 0)
        new_attempt_count = current_attempt_count + 1

        # Update state
        return {
            "components": state["components"],
            "generated_content": response.content,
            "validation_results": state.get("validation_results", {}),
            "errors": [],
            "attempt_count": new_attempt_count,
            "status": "generated",
            "generation_history": state.get("generation_history", []) + [{
                "attempt_number": new_attempt_count,
                "content": response.content,
                "timestamp": datetime.now().isoformat()
            }]
        }

    except Exception as e:
        # Handle errors by updating the state appropriately
        current_attempt_count = state.get("attempt_count", 0)
        return {
            "components": state["components"],
            "generated_content": "",
            "validation_results": state.get("validation_results", {}),
            "errors": [f"Generation error: {str(e)}"],
            "attempt_count": current_attempt_count,
            "status": "error",
            "generation_history": state.get("generation_history", [])
        }

def validate_content(state: State) -> State:
    """Validate the generated content."""
    try:
        component = state["components"][0]
        content = state["generated_content"].strip()

        # Basic validation using dictionary key access
        validation_results = {
            "char_count": len(content),
            "within_char_limit": len(content) <= component["char_limit"],
            "is_empty": not content
        }

        # Update state including attempt_count from the state (DO NOT increment here)
        return {
            "components": state["components"],
            "generated_content": state["generated_content"],
            "validation_results": validation_results,
            "errors": state.get("errors", []),
            "attempt_count": state["attempt_count"],  # Get count from the state
            "status": "completed" if validation_results["within_char_limit"] else "validation_failed",
            "generation_history": state.get("generation_history", [])
        }

    except Exception as e:
        # Handle errors by updating the state appropriately
        return {
            "components": state["components"],
            "generated_content": state["generated_content"],
            "validation_results": state["validation_results"],
            "errors": state.get("errors", []) + [f"Validation error: {str(e)}"],
            "attempt_count": state["attempt_count"],  # Get count from the state
            "status": "error",
            "generation_history": state.get("generation_history", [])
        }

def should_continue(state: State) -> str:
    """Determine if generation should continue."""
    if state["status"] == "completed":
        return END  # End if validation is successful
    if state["status"] == "validation_failed":
        if state["attempt_count"] < state["components"][0]["max_attempts"]:
            return "generate"  # Continue to generate if below max attempts
        else:
            state["status"] = "max_attempts_reached"
            return END  # End if max attempts reached and still failing validation
    return END  # Default to end if status is anything else
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

# Load the knowledge base
knowledge_base = load_knowledge_base("template_kb.json")

# Compile the graph
graph = workflow.compile()

# Example input for testing in LangGraph Studio
[{"component_type": "headline basic", "element_type": "Introduction Copy", "char_limit": 400, "token_limit": 200, "audience": "Schwimmen", "max_attempts": 3}]