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
    generated_content: str
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

async def generate_content(state: State) -> State:
    """
    Generates copy content using LLM based on the component specifications (async).

    Args:
        state (State): Current workflow state

    Returns:
        State: Updated state with generated content and attempt information

    Note:
        Uses retry logic for API calls and maintains generation history
    """
    knowledge_base = load_knowledge_base("template_kb.json")
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            max_retries=2,
            request_timeout=30
        )

        # Process first component (for demo purposes)
        component = state["components"][0]

        # Get relevant information from the knowledge base using dictionary key access
        kb_info = knowledge_base.get(component["component_type"], {}).get(component["element_type"], {})

        # Enhanced feedback logic
        feedback_text = ""
        if state.get("status") == "validation_failed" and state.get("validation_results"):
            validation_results = state["validation_results"]
            feedback_parts = []
            
            if not validation_results["within_char_limit"]:
                char_diff = validation_results["char_count"] - component["char_limit"]
                feedback_parts.append(f"- Der Text ist um {char_diff} Zeichen zu lang. Bitte kürze den Text entsprechend.")
            
            if validation_results.get("is_empty"):
                feedback_parts.append("- Der generierte Text war leer. Bitte generiere einen neuen Text.")
                
            if state.get("errors"):
                for error in state["errors"]:
                    feedback_parts.append(f"- {error}")
            
            if feedback_parts:
                feedback_text = "\n".join(feedback_parts)

        # Construct the prompt using the separate function
        prompt = construct_prompt(component, kb_info, feedback_text)

        response = await generate_with_retry(llm, prompt)

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
    """
    Validates the generated content against specified constraints.

    Args:
        state (State): Current workflow state containing generated content

    Returns:
        State: Updated state with validation results

    Validates:
        - Character count against limit
        - Content emptiness
    """
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
    """
    Determines if the generation process should continue based on current state.

    Args:
        state (State): Current workflow state

    Returns:
        str: Next action to take ('generate', END)

    Decision logic:
        - Ends if validation is successful
        - Continues generation if under max attempts
        - Ends if max attempts reached
    """
    if state["status"] == "completed":
        return END  # End if validation is successful
    if state["status"] == "validation_failed":
        if state["attempt_count"] < state["components"][0]["max_attempts"]:
            return "generate"  # Continue to generate if below max attempts
        else:
            state["status"] = "max_attempts_reached"
            return END  # End if max attempts reached and still failing validation
    return END  # Default to end if status is anything else

# Workflow Graph Setup
# Create a directed graph for the content generation workflow
workflow = StateGraph(State)

# Define the main processing nodes
workflow.add_node("generate", generate_content)  # Content generation node
workflow.add_node("validate", validate_content)  # Content validation node

# Define the workflow edges
workflow.add_edge(START, "generate")  # Start with generation
workflow.add_edge("generate", "validate")  # Validate after generation

# Add conditional logic for workflow continuation
workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        END: END,  # End workflow if complete or max attempts reached
        "generate": "generate"  # Retry generation if needed
    }
)

# Load the knowledge base
knowledge_base = load_knowledge_base("template_kb.json")

# Compile the graph
graph = workflow.compile()

