from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
from datetime import datetime
from pathlib import Path
from langchain_core.messages import BaseMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s)')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CopyComponent(BaseModel):
    name: str
    char_limit: int
    briefing: str
    audience: str
    component_type: str
    element_type: str

class InputSchema(BaseModel):
    components: List[CopyComponent]
    max_attempts: int = Field(default=3)

class State(BaseModel):
    input: InputSchema
    generated_content: List[Dict[str, str]] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    attempt_count: int = Field(default=0)
    status: str = Field(default="")
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    generation_history: List[Dict[str, Any]] = Field(default_factory=list)

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)

def generate_content(state: State) -> State:
    """Generate copy using LLM based on component requirements."""
    current_attempt = state.attempt_count + 1
    logger.info(f"Starting generation attempt #{current_attempt}")
    
    if current_attempt > state.input.max_attempts:
        state.status = f"stopped_max_attempts_reached_{state.input.max_attempts}"
        return state

    try:
        for component in state.input.components:
            # Build previous attempts feedback
            previous_attempts_feedback = ""
            if state.generation_history:
                previous_attempts_feedback = "\nVORHERIGE VERSUCHE UND FEEDBACK:\n"
                for attempt in state.generation_history:
                    if attempt.get('component_name') == component.name:
                        previous_attempts_feedback += f"""
Versuch #{attempt['attempt_number']}:
Text: {attempt['content']}
Feedback: {attempt['feedback']}
---"""

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

{previous_attempts_feedback}

Erstelle den Text in deutscher Sprache und beachte dabei das Feedback aus vorherigen Versuchen.
WICHTIG: Der Text MUSS EXAKT {component.char_limit} Zeichen lang sein."""

            response = generate_with_retry(ChatOpenAI(
                model="gpt-4",
                temperature=0.4,
                max_retries=3,
                request_timeout=30
            ), prompt)
            
            state.generated_content.append({
                "component_name": component.name,
                "content": response.content
            })
            state.messages.append({"role": "assistant", "content": response.content})
        
        state.attempt_count = current_attempt
        return state
        
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.status = "error_during_generation"
        return state

def validate_content(state: State) -> State:
    """Validate the generated content against requirements."""
    logger.info("Starting content validation")
    
    validation_results = []
    for component in state.input.components:
        content = next((item['content'] for item in state.generated_content 
                       if item['component_name'] == component.name), "")
        
        result = {
            "component_name": component.name,
            "char_count": len(content),
            "within_limit": len(content) == component.char_limit,
            "is_empty": not content.strip()
        }
        
        feedback = []
        if not result["within_limit"]:
            feedback.append(f"Character count mismatch. Expected {component.char_limit}, got {result['char_count']}.")
        if result["is_empty"]:
            feedback.append("Generated content is empty.")
            
        state.generation_history.append({
            "component_name": component.name,
            "attempt_number": state.attempt_count,
            "content": content,
            "feedback": " ".join(feedback) if feedback else "No issues found.",
            "validation_results": result
        })
        
        validation_results.append(result)
    
    all_valid = all(result["within_limit"] and not result["is_empty"] 
                    for result in validation_results)
    
    state.status = "completed" if all_valid else "validation_failed"
    return state

def export_results(state: State) -> State:
    """Export the generated content and metadata to a JSON file."""
    logger.info("Exporting results to JSON")
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "status": state.status,
        "total_attempts": state.attempt_count,
        "components": [
            {
                "name": comp.name,
                "content": next((item['content'] for item in state.generated_content 
                               if item['component_name'] == comp.name), ""),
                "validation": next((item['validation_results'] for item in state.generation_history 
                                  if item['component_name'] == comp.name), {})
            }
            for comp in state.input.components
        ],
        "generation_history": state.generation_history
    }
    
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"copy_export_{timestamp}.json"
    export_path = export_dir / filename
    
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results exported to {export_path}")
    return state

def should_continue(state: State) -> str:
    """Determine if the process should continue or end."""
    if state.status == "completed":
        return "export"
    elif state.status == "validation_failed" and state.attempt_count < state.input.max_attempts:
        return "generate"
    else:
        return "export"

# Create the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate", generate_content)
workflow.add_node("validate", validate_content)
workflow.add_node("export", export_results)

# Add edges
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")

# Add conditional edges
workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        "generate": "generate",
        "export": "export"
    }
)

workflow.add_edge("export", END)

# Compile the graph
graph = workflow.compile()