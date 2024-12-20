
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
    components: List[CopyComponent]  # Now consistently using CopyComponent
    generated_content: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]
    output: List[Dict[str, Any]]

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

def construct_prompt(component: CopyComponent, kb_info: Dict[str, Any], feedback_text: str) -> str:
    """Constructs the prompt for the LLM based on component and knowledge base information."""
    examples_text = ""
    if kb_info:
        examples_text = "\nBEISPIELE ZUR ORIENTIERUNG:\n"
        for ex in kb_info.get("examples", [])[:2]:
            examples_text += f"Beispiel Output ({len(ex.get('expected_output', ''))} Zeichen): {ex.get('expected_output', '')}\n"

    char_limit = kb_info.get('char_limit', component.char_limit)

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
- Der Content MUSS spezifisch auf {component.audience} ausgerichtet sein
- STRENGE Längenbegrenzung:
  - Maximal {char_limit} Zeichen (inkl. Leerzeichen und Satzzeichen)
- Keine direkte Kundenanrede
- Keine CTAs oder Verlinkungen im Einführungstext
- Keine Anführungszeichen verwenden

TECHNISCHE ANFORDERUNGEN:
- Zielgruppe: {component.audience}
- Komponententyp: {component.component_type}
- Elementtyp: {component.element_type}

FORMATIERUNG:
- Gib NUR den reinen Text zurück, ohne Präfixe wie 'Text:' oder 'Content:'
- Keine zusätzliche Formatierung
- Text muss in deutscher Sprache sein

{feedback_text}

WICHTIG: Der Text MUSS kürzer als {char_limit} Zeichen sein. Zähle jeden Buchstaben, jedes Leerzeichen und jedes Satzzeichen."""
    return prompt

def generate_feedback(validation_results: Dict[str, Any], component: CopyComponent) -> List[str]:
    """Generate detailed feedback messages based on validation results."""
    feedback_messages = []

    # Character limit validation - access char_limit from component
    if not validation_results.get("within_char_limit", True):
        char_count = validation_results.get("char_count", 0)
        char_limit = component.char_limit  # Get char_limit from component
        char_diff = char_count - char_limit
        feedback_messages.append(
            f"Zeichenlimit überschritten: {char_count}/{char_limit} "
            f"({char_diff} Zeichen zu lang)"
        )

    # Token limit validation - access token_limit from component
    if not validation_results.get("within_token_limit", True):
        token_count = validation_results.get("token_count", 0)
        token_limit = component.token_limit  # Get token_limit from component
        token_diff = token_count - token_limit
        feedback_messages.append(
            f"Token-Limit überschritten: {token_count}/{token_limit} "
            f"({token_diff} Token zu lang)"
        )

    # Empty content validation
    if validation_results.get("is_empty", False):
        feedback_messages.append("Der generierte Text ist leer")

    # Content quality checks
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
    state["generated_content"] = state.get("generated_content", [])
    state["validation_results"] = state.get("validation_results", [])
    state["errors"] = state.get("errors", [])
    state["attempt_count"] = state.get("attempt_count", 0) + 1
    state["status"] = "generating"
    state["generation_history"] = state.get("generation_history", [])

    # Convert dictionaries to CopyComponent objects
    if not all(isinstance(comp, CopyComponent) for comp in state["components"]):
        state["components"] = [CopyComponent(**comp) for comp in state["components"]]

    knowledge_base = load_knowledge_base("template_kb.json")

    generated_contents = []
    for component in state["components"]:
        try:
            llm = ChatOpenAI(
                model="gpt-4-turbo-preview",
                temperature=0.2,
                max_retries=2,
                request_timeout=30
            )

            kb_info = knowledge_base.get(component.component_type, {}).get(component.element_type, {})

            # Enhanced feedback logic
            feedback_text = ""
            if state.get("status") == "validation_failed" and state.get("errors"):
                # Find feedback relevant to the current component
                relevant_feedback = [
                    err for err in state["errors"]
                    if err.startswith(f"{component.element_type}:")
                ]
                if relevant_feedback:
                    feedback_text = "\n".join(relevant_feedback)

            prompt = construct_prompt(component, kb_info, feedback_text)
            response = await generate_with_retry(llm, prompt)

            # Add the generated content with component info
            generated_contents.append({
                "component_type": component.component_type,
                "element_type": component.element_type,
                "content": response.content,
                "timestamp": datetime.now().isoformat()
            })

        except Exception as e:
            state["errors"].append(f"Generation error for {component.element_type}: {str(e)}")
            state["status"] = "error"
            return state # Exit if one component fails

    state["generated_content"] = generated_contents
    state["status"] = "content_generated"
    return state

def validate_content(state: State) -> State:
    """Validates the generated content against specified criteria."""
    all_validation_results = []
    all_errors = []

    # Validate each component's content
    for idx, component in enumerate(state["components"]):
        if idx >= len(state["generated_content"]):
            logger.warning(f"No generated content found for component: {component.element_type}")
            continue

        content = clean_content(state["generated_content"][idx]["content"])

        validation_results = {
            "component_type": component.component_type,
            "element_type": component.element_type,
            "char_count": len(content),
            "within_char_limit": len(content) <= component.char_limit,
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
            all_errors.extend([f"{component.element_type}: {msg}" for msg in feedback_messages])

        all_validation_results.append(validation_results)

    # Update state with all results
    state["validation_results"] = all_validation_results
    state["errors"] = all_errors

    # Update status based on overall validation
    if all_errors:
        state["status"] = "validation_failed"
    else:
        state["status"] = "validation_passed"

    return state

def should_continue(state: State) -> str:
    """Determines if the generation process should continue."""
    if state["status"] == "validation_passed":
        return "format_output"

    # Check if any component has reached max attempts
    if state.get("attempt_count", 0) >= state["components"][0].max_attempts if state["components"] else False:
        state["status"] = "max_attempts_reached"
        return END

    if state["status"] == "validation_failed":
        return "generate"

    return END

def format_output(state: State) -> State:
    """Formats the final output as a clean JSON structure."""
    logger.info(f"Entering format_output with state: {state}")
    output = []

    for idx, component in enumerate(state["components"]):
        logger.info(f"Processing component at index {idx}: {component}")
        result = {
            "component_type": component.component_type,
            "element_type": component.element_type,
            "audience": component.audience
        }

        # Check if we have validation results for this component
        if idx < len(state["validation_results"]):
            validation = state["validation_results"][idx]

            if validation.get("within_char_limit", True) and not validation.get("is_empty", False):
                # If validation passed, include the content
                result["status"] = "success"
                result["content"] = state["generated_content"][idx]["content"]
            else:
                # If validation failed, include the error messages
                result["status"] = "failed"
                errors = [msg for msg in state["errors"] if msg.startswith(f"{component.element_type}:")]
                result["errors"] = errors if errors else ["Unknown validation error"]
        else:
            # Component wasn't processed
            result["status"] = "error"
            result["errors"] = ["Component processing failed"]

        output.append(result)

    state["output"] = output
    return state

# Workflow Graph Setup
workflow = StateGraph(State)

# Define the main processing nodes
workflow.add_node("generate", generate_content)
workflow.add_node("validate", validate_content)
workflow.add_node("format_output", format_output)

# Define the workflow edges
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "validate")
workflow.add_conditional_edges(
    "validate",
    should_continue,
    {
        "format_output": "format_output",
        END: END,
        "generate": "generate"
    }
)
workflow.add_edge("format_output", END)

# Compile the graph
graph = workflow.compile()