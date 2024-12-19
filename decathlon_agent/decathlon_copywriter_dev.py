"""
Decathlon CRM Copywriter POC using LangGraph.
This script implements a graph-based workflow for generating and validating marketing copy.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
import logging
from tenacity import retry, wait_exponential, stop_after_attempt
import tiktoken
import math
import time

# Add knowledge base import
from knowledge_base.template_kb import get_template_kb, TemplateKnowledgeBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass(frozen=True)
class CopyComponent:
    component_type: str
    element_type: str
    char_limit: int
    token_limit: int
    audience: str
    max_attempts: int = 3

class AgentState(TypedDict):
    """Main state class for the copywriting workflow."""
    component: CopyComponent
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]

def create_initial_state(component: CopyComponent) -> AgentState:
    """Create initial state with proper typing."""
    return AgentState(
        component=component,
        generated_content="",
        validation_results={},
        errors=[],
        attempt_count=0,
        status="initialized",
        generation_history=[]
    )

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Count the number of tokens in a given text for a specific model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model}: {str(e)}. Falling back to character-based estimation.")
        return estimate_tokens_by_char_count(text)

def estimate_tokens_by_char_count(text: str) -> int:
    """Estimate token count based on character length."""
    return math.ceil(len(text) / 4)  # Roughly 4 characters per token on average

@retry(wait=wait_exponential(multiplier=1, min=1, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)

def validate_limits(content: str, char_limit: int, token_limit: int) -> Dict[str, Any]:
    """Validation for character and token limits."""
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
    return cleaned.strip('"')

def generate_feedback(validation_results: Dict[str, Any], char_limit: int, token_limit: int) -> List[str]:
    """Generate feedback based on validation results."""
    feedback = []
    
    # Check for issues
    if not validation_results['within_char_limit']:
        excess_chars = validation_results['char_count'] - char_limit
        feedback.append(f"Der Text ist zu lang ({validation_results['char_count']} Zeichen). "
                        f"Bitte kürze um ca. {excess_chars} Zeichen.")
    if not validation_results['within_token_limit']:
        excess_tokens = validation_results['token_count'] - token_limit
        feedback.append(f"Der Text hat zu viele Tokens ({validation_results['token_count']} Tokens). "
                        f"Bitte kürze um ca. {excess_tokens} Tokens.")
    if validation_results['is_empty']:
        feedback.append("Der generierte Text ist leer.")
        
    # Add positive feedback for successful generation
    if (validation_results['within_char_limit'] and 
        validation_results['within_token_limit'] and 
        not validation_results['is_empty']):
        feedback.append("Text erfolgreich generiert und validiert.")
    
    return feedback

def update_state(
    current_state: AgentState,
    content: str,
    validation_results: dict,
    feedback: List[str],
    status: str
) -> AgentState:
    """
    Creates a new state instance with updated values.
    """
    return {
        **current_state,
        'generated_content': content,
        'validation_results': validation_results,
        'attempt_count': current_state['attempt_count'] + 1,
        'generation_history': [
            *current_state['generation_history'],
            {
                'attempt_number': current_state['attempt_count'] + 1,
                'content': content,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            }
        ],
        'status': status
    }

def generate_content(state: AgentState) -> AgentState:
    """Generate copy using LLM based on component requirements."""
    current_attempt = state['attempt_count'] + 1
    logger.info(f"Starting generation attempt #{current_attempt}")

    try:
        # Get knowledge base
        kb: TemplateKnowledgeBase = get_template_kb()

        # Get component attributes
        component = state['component']
        component_type = component.component_type if hasattr(component, 'component_type') else component['component_type']
        element_type = component.element_type if hasattr(component, 'element_type') else component['element_type']
        char_limit = component.char_limit if hasattr(component, 'char_limit') else component['char_limit']
        token_limit = component.token_limit if hasattr(component, 'token_limit') else component['token_limit']
        audience = component.audience if hasattr(component, 'audience') else component['audience']

        template = kb.get_template(component_type, element_type)

        # Get examples from template
        examples_text = ""
        if template and hasattr(template, 'examples') and template.examples:
            examples_text = "\nBEISPIELE ZUR ORIENTIERUNG:\n"
            for ex in template.examples[:2]:
                examples_text += f"Beispiel Output ({len(ex.expected_output)} Zeichen): {ex.expected_output}\n"

        # Build previous attempts feedback
        previous_attempts_feedback = ""
        if state['generation_history']:
            previous_attempts_feedback = "\nVORHERIGE VERSUCHE UND FEEDBACK:"
            for attempt in state['generation_history']:
                previous_attempts_feedback += f"""
Versuch #{attempt['attempt_number']}:
Content: {attempt['content']} ({len(attempt['content'])} Zeichen)
Feedback: {attempt['feedback']}
---"""

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

        # Construct prompt
        prompt = f"""Du bist ein erfahrener Decathlon CRM Copywriter, spezialisiert auf die Erstellung von E-Mail-Inhalten.

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
- Der Content MUSS spezifisch auf {audience} ausgerichtet sein
- STRENGE Längenbegrenzung:
  - Maximal {char_limit} Zeichen (inkl. Leerzeichen und Satzzeichen)
  - Maximal {token_limit} Tokens
- Keine direkte Kundenanrede
- Keine CTAs oder Verlinkungen im Einführungstext
- Keine Anführungszeichen verwenden

TECHNISCHE ANFORDERUNGEN:
- Zielgruppe: {audience}
- Komponententyp: {component_type}
- Elementtyp: {element_type}

FORMATIERUNG:
- Gib NUR den reinen Text zurück, ohne Präfixe wie 'Text:' oder 'Content:'
- Keine zusätzliche Formatierung
- Text muss in deutscher Sprache sein

{previous_attempts_feedback}

WICHTIG: Der Text MUSS kürzer als {char_limit} Zeichen und {token_limit} Tokens sein. Zähle jeden Buchstaben, jedes Leerzeichen und jedes Satzzeichen."""

        # Add backoff between attempts
        time.sleep(current_attempt * 2)

        # Generate content
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            max_retries=3,
            request_timeout=30
        )
        response = generate_with_retry(llm, prompt)

        # Clean and validate the generated content
        new_content = clean_content(response.content)
        validation_results = validate_limits(
            new_content,
            char_limit,
            token_limit
        )

        # Log generation results
        logger.info(f"Generated content with {validation_results['char_count']} characters "
                   f"and {validation_results['token_count']} tokens")

        # Update and return new state
        return update_state(
            current_state=state,
            content=new_content,
            validation_results=validation_results,
            feedback=[],  # Feedback will be generated in validation step
            status="generated"
        )

    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        return update_state(
            current_state=state,
            content=state['generated_content'],
            validation_results=state['validation_results'],
            feedback=[error_msg],
            status="error_during_generation"
        )

def validate_content(state: AgentState) -> AgentState:
    """Validate the generated content against requirements."""
    logger.info("Starting content validation")
    
    # Get current content
    current_content = clean_content(state['generated_content'])
    
    # Check for duplicates only if we have previous attempts
    previous_contents = [
        attempt['content'] 
        for attempt in state['generation_history'][:-1]  # Exclude current attempt
    ] if len(state['generation_history']) > 1 else []
    
    is_duplicate = current_content in previous_contents
    
    # Validate content
    validation_results = validate_limits(
        current_content,
        state['component'].char_limit,
        state['component'].token_limit
    )
    
    # Generate feedback
    feedback = []
    
    # Check validation first
    is_valid = (
        validation_results['within_char_limit'] and 
        validation_results['within_token_limit'] and 
        not validation_results['is_empty']
    )
    
    # Only add duplicate feedback if it's actually a duplicate and not the first attempt
    if is_duplicate and len(state['generation_history']) > 1:
        feedback.append("Identischer Text wurde bereits generiert.")
    
    if not is_valid:
        feedback.extend(generate_feedback(
            validation_results,
            state['component'].char_limit,
            state['component'].token_limit
        ))
    else:
        feedback.append("Text erfolgreich generiert und validiert.")
    
    # Update history
    updated_history = state['generation_history'].copy()
    if updated_history:
        updated_history[-1] = {
            **updated_history[-1],
            'feedback': feedback,
            'validation_results': validation_results
        }
    
    # Set status
    new_status = "completed" if is_valid else "validation_failed"
    
    return {
        **state,
        'generated_content': current_content,
        'validation_results': validation_results,
        'errors': feedback,
        'generation_history': updated_history,
        'status': new_status
    }

def should_regenerate(state: AgentState) -> str:
    """Determine if content should be regenerated."""
    # Check if we've reached max attempts
    if state['attempt_count'] >= state['component'].max_attempts:
        logger.info(f"Reached maximum attempts ({state['component'].max_attempts})")
        return END

    # Check current validation status
    validation_results = state['validation_results']
    if (validation_results.get('within_char_limit', False) and 
        validation_results.get('within_token_limit', False) and 
        not validation_results.get('is_empty', True)):
        logger.info("Content meets all requirements, proceeding to export")
        return END

    # Check if status is explicitly marked as completed
    if state['status'] == "completed":
        return END
        
    # Continue generation if validation failed and we haven't reached max attempts
    if state['status'] == "validation_failed":
        return "generate"
        
    return END

def export_results(state: AgentState) -> AgentState:
    """Export the generated content and metadata to a JSON file."""
    logger.info("Exporting results to JSON")

    export_data = {
        "timestamp": datetime.now().isoformat(),
        "component": {
            "component_type": state['component'].component_type,
            "element_type": state['component'].element_type,
            "audience": state['component'].audience,
            "char_limit": state['component'].char_limit,
            "token_limit": state['component'].token_limit
        },
        "generation_result": {
            "final_content": state['generated_content'],
            "char_count": state['validation_results']['char_count'],
            "token_count": state['validation_results']['token_count'],
            "status": state['status'],
            "total_attempts": state['attempt_count']
        },
        "generation_history": state['generation_history']
    }

    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"copy_export_{timestamp}_{state['component'].component_type.replace(' ', '_')}_{state['component'].element_type.replace(' ', '_')}.json"
    export_path = export_dir / filename

    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Results exported to {export_path}")

    return state

class DecathlonCopywriterAgent:
    def __init__(self):
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        """Create the workflow graph with proper state schema."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("generate", generate_content)
        workflow.add_node("validate", validate_content)
        workflow.add_node("export", export_results)

        # Define edges
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "validate")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "validate",
            should_regenerate,
            {
                END: "export",
                "generate": "generate"
            }
        )

        workflow.add_edge("export", END)

        return workflow.compile()

    def generate_copy(self, component: CopyComponent) -> AgentState:
        """Public interface for generating copy."""
        logger.info(f"Generating copy for component: {component.component_type} - {component.element_type}")
        initial_state = create_initial_state(component)
        return self.workflow.invoke(initial_state)

# Example usage
if __name__ == "__main__":
    agent = DecathlonCopywriterAgent()

    test_components = [
        CopyComponent(
            component_type="headline basic",
            element_type="Introduction Copy",
            char_limit=400,
            token_limit=200,
            audience="Schwimmen"
        ),
        CopyComponent(
            component_type="advice",
            element_type="Content Title",
            char_limit=30,
            token_limit=15,
            audience="Schwimmen"
        )
    ]

    for component in test_components:
        try:
            print(f"\n{'='*80}")
            print(f"Testing component: {component.component_type} - {component.element_type}")
            print(f"Character limit: {component.char_limit}")
            print(f"Token limit: {component.token_limit}")
            print(f"{'='*80}")

            result = agent.generate_copy(component)

            print("\nGeneration Results:")
            print(f"Status: {result['status']}")
            print(f"Total attempts: {result['attempt_count']}")
            
            if result['generated_content']:
                print(f"\nFinal Content:")
                print(f"Length: {len(result['generated_content'])} chars, "
                      f"{result['validation_results'].get('token_count', 0)} tokens")
                print(f"Content: {result['generated_content']}\n")
                
                if result['validation_results']:
                    print("Validation Results:")
                    for key, value in result['validation_results'].items():
                        print(f"{key}: {value}")
            
            if result['generation_history']:
                print("\nGeneration History:")
                for attempt in result['generation_history']:
                    print(f"\nAttempt #{attempt['attempt_number']}:")
                    print(f"Content: {attempt['content']}")
                    # Safely access feedback
                    if 'feedback' in attempt:
                        print(f"Feedback: {', '.join(attempt['feedback']) if attempt['feedback'] else 'No feedback'}")

            time.sleep(5)  # Add cooldown between components

        except Exception as e:
            logger.error(f"Error processing component {component.component_type} - {component.element_type}: {str(e)}")
            continue