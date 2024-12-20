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
from langchain.schema import SystemMessage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Schema definitions
@dataclass
class CopyComponent:
    """
    Represents a content component for copy generation.

    Attributes:
        component_type (str): Type of the component (e.g., 'headline basic')
        element_type (str): Specific element within the component type
        char_limit (int): Maximum number of characters allowed
        token_limit (int): Maximum number of tokens allowed
        audience (str): Target audience for the content
        max_attempts (int): Maximum number of generation attempts allowed
    """
    component_type: str
    element_type: str
    char_limit: int
    token_limit: int
    audience: str
    max_attempts: int = 3

class State(TypedDict):
    components: List[CopyComponent]
    generated_content: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]

# Knowledge base loading function (remains synchronous as it's a one-time load)
def load_knowledge_base(filepath: str) -> dict:
    """Loads the knowledge base from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

# Reuse your existing functions with modifications for Studio compatibility
@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
async def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic (async)."""
    return await llm.ainvoke(prompt)

def construct_prompt(component: Dict[str, Any], kb_info: Dict[str, Any], feedback_text: str) -> str:
    """Constructs the prompt for the LLM based on component and knowledge base information."""
    examples_text = ""
    if kb_info:
        examples_text = "\nBEISPIELE ZUR ORIENTIERUNG:\n"
        for ex in kb_info.get("examples", [])[:2]:
            examples_text += f"Beispiel Output ({len(ex.get('expected_output', ''))} Zeichen): {ex.get('expected_output', '')}\n"

    char_limit = kb_info.get('char_limit', component["char_limit"])

    # Determine length guidance based on char_limit
    length_guidance = """
LÄNGENHINWEISE:
- Nutze kurze, prägnante Wörter
- Vermeide Füllwörter und lange Komposita
- Jedes Wort muss einen Mehrwert bringen
- Nutze Ausrufezeichen sparsam
""" if char_limit <= 35 else """
LÄNGENHINWEISE:
- Halte die Sätze kurz und prägnant
- Vermeide Füllwörter und Wiederholungen
- Jeder Satz sollte einen klaren Mehrwert haben
"""

    # Add feedback section if provided
    feedback_section = ""
    if feedback_text:
        feedback_section = f"""
FEEDBACK ZUM VORHERIGEN VERSUCH:
{feedback_text}
Bitte berücksichtige dieses Feedback bei der Generierung des neuen Textes.
"""

    prompt = f"""Du bist ein erfahrener Decathlon CRM Copywriter, spezialisiert auf die Erstellung von E-Mail-Inhalten.

{feedback_section}

ROLLE UND AUFGABE:
- Entwickle inspirierende, energiegeladene Kommunikation für sportbegeisterte Menschen
- Schaffe eine persönliche, einladende Atmosphäre
- Stelle den Markenkern "Sport für alle zugänglich und erfreulich zu machen" in den Mittelpunkt
- Fördere eine positive, kundenorientierte Beziehung

MARKENIDENTITÄT UND TONALITÄT:
- Freundlich und Einladend: Persönliche, einladende Note in jeder Nachricht
- Begeistert und Positiv: Vermittle die Freude am Sport und Outdoor-Aktivitäten
- Kundenorientiert und Unterstützend: Biete inspirierende Inhalte und klare Orientierung
- Einfach und Direkt: Vermeide Fachjargon, halte die Sprache verständlich
- Spielerisch und Energetisch: Nutze leichten Humor und Dynamik wo angebracht

{length_guidance}

BEISPIELE ZUR ORIENTIERUNG:
{examples_text}

WICHTIGE REGELN:
- Der Content MUSS spezifisch auf {component["audience"]} ausgerichtet sein
- STRENGE Längenbegrenzung:
  - Maximal {char_limit} Zeichen (inkl. Leerzeichen und Satzzeichen)
- Keine direkte Kundenanrede
- Keine CTAs oder Verlinkungen im Einführungstext
- Keine Anführungszeichen verwenden

TECHNISCHE ANFORDERUNGEN:
- Zielgruppe: {component["audience"]}
- Komponententyp: {component["component_type"]}
- Elementtyp: {component["element_type"]}

FORMATIERUNG:
- Gib NUR den reinen Text zurück, ohne Präfixe wie 'Text:' oder 'Content:'
- Keine zusätzliche Formatierung
- Text muss in deutscher Sprache sein

{feedback_text}

WICHTIG: Der Text MUSS kürzer als {char_limit} Zeichen sein. Zähle jeden Buchstaben, jedes Leerzeichen und jedes Satzzeichen."""
    return prompt

def generate_feedback(validation_results: Dict[str, Any], component: Dict[str, Any]) -> List[str]:
    """Generate detailed feedback messages based on validation results."""
    feedback_messages = []
    
    # Character limit validation
    if not validation_results.get("within_char_limit", True):
        char_count = validation_results.get("char_count", 0)
        char_diff = char_count - component["char_limit"]
        feedback_messages.append(
            f"Zeichenlimit überschritten: {char_count}/{component['char_limit']} "
            f"({char_diff} Zeichen zu lang)"
        )
    
    # Token limit validation
    if not validation_results.get("within_token_limit", True):
        token_count = validation_results.get("token_count", 0)
        token_diff = token_count - component["token_limit"]
        feedback_messages.append(
            f"Token-Limit überschritten: {token_count}/{component['token_limit']} "
            f"({token_diff} Token zu lang)"
        )
    
    # Empty content validation
    if validation_results.get("is_empty", False):
        feedback_messages.append("Der generierte Text ist leer")
    
    # Content quality checks (can be expanded)
    if validation_results.get("has_unwanted_prefixes", False):
        feedback_messages.append("Text enthält unerwünschte Präfixe (z.B. 'Text:', 'Content:')")
    
    return feedback_messages

def clean_content(content: str) -> str:
    """Clean the content by removing unwanted prefixes and formatting."""
    # List of common prefixes to remove
    prefixes = ["text:", "content:", "output:", "generated text:"]
    
    # Convert to lowercase for case-insensitive comparison
    content_lower = content.lower()
    
    # Remove known prefixes
    for prefix in prefixes:
        if content_lower.startswith(prefix):
            content = content[len(prefix):].strip()
            break
    
    # Remove any leading/trailing whitespace
    content = content.strip()
    
    return content

async def generate_content(state: State) -> State:
    """Generates copy content using LLM based on the component specifications (async)."""
    # Initialize state fields if they don't exist
    if "generated_content" not in state:
        state["generated_content"] = []
    if "validation_results" not in state:
        state["validation_results"] = {}
    if "errors" not in state:
        state["errors"] = []
    if "attempt_count" not in state:
        state["attempt_count"] = 0
    if "status" not in state:
        state["status"] = "started"
    if "generation_history" not in state:
        state["generation_history"] = []

    knowledge_base = load_knowledge_base("template_kb.json")
    
    try:
        llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0.2,
            max_retries=2,
            request_timeout=30
        )

        # Process all components instead of just the first one
        for component in state["components"]:
            kb_info = knowledge_base.get(component["component_type"], {}).get(component["element_type"], {})

            # Enhanced feedback logic
            feedback_text = ""
            if state.get("status") == "validation_failed" and state.get("validation_results"):
                validation_results = state["validation_results"]
                feedback_parts = []
                
                if not validation_results["within_char_limit"]:
                    char_diff = validation_results["char_count"] - component["char_limit"]
                    feedback_parts.append(f"- Der Text ist um {char_diff} Zeichen zu lang")
                
                if validation_results.get("is_empty"):
                    feedback_parts.append("- Der generierte Text war leer")
                    
                if state.get("errors"):
                    for error in state["errors"]:
                        feedback_parts.append(f"- {error}")
                
                if feedback_parts:
                    feedback_text = "\n".join(feedback_parts)

            prompt = construct_prompt(component, kb_info, feedback_text)
            response = await generate_with_retry(llm, prompt)

            # Add the generated content with component info
            state["generated_content"].append({
                "component_type": component["component_type"],
                "element_type": component["element_type"],
                "content": response.content,
                "timestamp": datetime.now().isoformat()
            })

        return state

    except Exception as e:
        state["errors"] = [f"Generation error: {str(e)}"]
        state["status"] = "error"
        return state

def validate_content(state: State) -> State:
    """Validates the generated content against specified criteria."""
    all_validation_results = []
    all_errors = []
    
    # Validate each component's content
    for idx, component in enumerate(state["components"]):
        if idx >= len(state["generated_content"]):
            continue
            
        content = clean_content(state["generated_content"][idx]["content"])
        
        validation_results = {
            "component_type": component["component_type"],
            "element_type": component["element_type"],
            "char_count": len(content),
            "within_char_limit": len(content) <= component["char_limit"],
            "is_empty": len(content.strip()) == 0,
            "has_unwanted_prefixes": any(
                content.lower().startswith(prefix) 
                for prefix in ["text:", "content:", "output:"]
            ),
        }
        
        # Update the cleaned content
        state["generated_content"][idx]["content"] = content
        
        # Generate feedback for this component
        feedback_messages = generate_feedback(validation_results, component)
        if feedback_messages:
            all_errors.extend([f"{component['element_type']}: {msg}" for msg in feedback_messages])
            
        all_validation_results.append(validation_results)
    
    # Update state with all results
    state["validation_results"] = all_validation_results
    state["errors"] = all_errors
    
    # Update status based on overall validation
    if all_errors:
        state["status"] = "validation_failed"
    else:
        state["status"] = "completed"
    
    return state

def should_continue(state: State) -> str:
    """Determines if the generation process should continue."""
    if state["status"] == "completed":
        return END
        
    # Check if any component has reached max attempts
    all_max_attempts_reached = all(
        state.get("attempt_count", 0) >= component["max_attempts"]
        for component in state["components"]
    )
    
    if all_max_attempts_reached:
        state["status"] = "max_attempts_reached"
        return END
        
    if state["status"] == "validation_failed":
        return "generate"
        
    return END

# Workflow Graph Setup
workflow = StateGraph(State)

# Define the main processing nodes
workflow.add_node("generate", generate_content)
workflow.add_node("validate", validate_content)

# Define the workflow edges
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")

# Add conditional logic for workflow continuation
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

