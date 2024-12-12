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
import json
import tiktoken
import math
import time


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
    token_limit: int = Field(default=45)

class ValidationResult(BaseModel):
    char_count: int
    token_count: int
    within_char_limit: bool
    within_token_limit: bool
    is_empty: bool

class GenerationAttempt(BaseModel):
    attempt_number: int
    content: str
    feedback: str
    validation_results: ValidationResult

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


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a given text."""
    encoding = tiktoken.encoding_for_model(model)
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model}: {str(e)}. Falling back to character-based estimation.")
        return estimate_tokens_by_char_count(text)


def estimate_tokens_by_char_count(text: str) -> int:
    """Estimate token count based on character length."""
    return math.ceil(len(text) / 3)


def validate_limits(content: str, char_limit: int, token_limit: int) -> Dict[str, Any]:
    """Centralized validation logic for both character and token limits."""
    char_count = len(content)
    token_count = count_tokens(content)
    return {
        "char_count": char_count,
        "token_count": token_count,
        "within_char_limit": char_count <= char_limit,
        "within_token_limit": token_count <= token_limit,
        "is_empty": not content.strip()
    }


def clean_content(content: str) -> str:
    """Centralized content cleaning logic."""
    cleaned = content.strip()
    prefixes = ['Text:', 'Content:', 'Text: ', 'Content: ']
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    return cleaned.strip('"\'')


def generate_feedback(validation_results: Dict[str, Any], char_limit: int, token_limit: int) -> List[str]:
    """Centralized feedback generation."""
    feedback = []
    if not validation_results['within_char_limit']:
        excess_chars = validation_results['char_count'] - char_limit
        feedback.append(f"Der Text ist zu lang ({validation_results['char_count']} Zeichen). "
                       f"Bitte kürze um ca. {excess_chars} Zeichen.")
    elif not validation_results['within_token_limit']:
        feedback.append(f"Hinweis: Token-Optimierung möglich ({validation_results['token_count']}/{token_limit})")
    if validation_results['is_empty']:
        feedback.append("Der generierte Text ist leer.")
    return feedback


@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)


def generate_content(state: State) -> State:
    """Generate copy using LLM based on component requirements."""
    current_attempt = state.attempt_count + 1
    logger.info(f"Starting generation attempt #{current_attempt}")
    
    try:
        # Process each component
        for component in state.input.components:
            # Check for max attempts first
            if current_attempt > state.input.max_attempts:
                state.status = f"stopped_max_attempts_reached_{state.input.max_attempts}"
                return state
            
            # Build previous attempts feedback
            previous_attempts_feedback = ""
            if state.generation_history:
                previous_attempts_feedback = "\nVORHERIGE VERSUCHE UND FEEDBACK:\n"
                for attempt in state.generation_history:
                    if attempt.get('component_name') == component.name:
                        previous_attempts_feedback += f"""
Versuch #{attempt['attempt_number']}:
Text: {attempt['content']} (Zeichen: {len(attempt['content'])})
Feedback: {attempt['feedback']}
---"""

            # Determine length guidance based on component type
            length_guidance = """
LÄNGENHINWEISE:
- Nutze kurze, prägnante Wörter
- Vermeide Füllwörter und lange Komposita
- Jedes Zeichen zählt, auch Leerzeichen
- Nutze Ausrufezeichen sparsam
""" if component.char_limit <= 35 else """
LÄNGENHINWEISE:
- Halte die Sätze kurz und prägnant
- Vermeide Füllwörter und Wiederholungen
- Jeder Satz sollte einen klaren Mehrwert haben
"""

            # Construct prompt
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
- STRENGE Längenbegrenzung: EXAKT {component.char_limit} Zeichen
- Der Content muss spezifisch auf {component.audience} ausgerichtet sein
{length_guidance}

{previous_attempts_feedback}

Erstelle den Text in deutscher Sprache und beachte dabei das Feedback aus vorherigen Versuchen.
WICHTIG: Der Text MUSS ABSOLUT GENAU {component.char_limit} Zeichen lang sein. Zähle JEDEN Buchstaben, jedes Leerzeichen und jedes Satzzeichen."""

            # Add backoff between attempts
            time.sleep(current_attempt * 2)
            
            # Generate content
            response = generate_with_retry(ChatOpenAI(
                model="gpt-4",
                temperature=0.4,
                max_retries=3,
                request_timeout=30
            ), prompt)
            
            # Clean and validate the generated content
            new_content = clean_content(response.content)
            validation_results = validate_limits(
                new_content,
                component.char_limit,
                component.token_limit
            )
            
            # Log generation results
            logger.info(f"Generated content with {validation_results['char_count']} characters "
                       f"and {validation_results['token_count']} tokens")
            
            # Add to generated content
            state.generated_content.append({
                "component_name": component.name,
                "content": new_content
            })
            
            # Add to messages
            state.messages.append({
                "role": "assistant",
                "content": new_content
            })
            
        state.attempt_count = current_attempt
        state.status = "generation_in_progress"
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
    
    for component in state.input.components:
        # Get content for this component
        content = next((item['content'] for item in state.generated_content 
                       if item['component_name'] == component.name), "")
        
        # Clean the content
        cleaned_content = clean_content(content)
        
        # Validate limits
        validation_results = validate_limits(
            cleaned_content,
            component.char_limit,
            component.token_limit
        )
        
        # Log validation results
        logger.info(f"Token count: {validation_results['token_count']}/{component.token_limit}")
        logger.info(f"Character count: {validation_results['char_count']}/{component.char_limit}")
        
        # Generate feedback
        feedback = generate_feedback(
            validation_results,
            component.char_limit,
            component.token_limit
        )
        
        # Add to generation history
        state.generation_history.append({
            "component_name": component.name,
            "attempt_number": state.attempt_count,
            "content": cleaned_content,
            "feedback": " ".join(feedback) if feedback else "Keine Probleme gefunden.",
            "validation_results": validation_results
        })
        
    # Update status based on validation results
    all_valid = all(
        attempt['validation_results']['within_char_limit'] 
        and not attempt['validation_results']['is_empty']
        for attempt in state.generation_history[-len(state.input.components):]
    )
    
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