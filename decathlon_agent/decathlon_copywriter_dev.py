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
from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
import time

# Add knowledge base import
from knowledge_base.template_kb import get_template_kb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass(frozen=True)
class CopyComponent:
    name: str  # Should correspond to 'module_element' in briefings
    token_limit: int
    char_limit: int
    audience: str
    component_type: str
    element_type: str
    max_attempts: int = 3
    url: Optional[str] = None

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

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count the number of tokens in a given text for a specific model."""
    encoding = tiktoken.encoding_for_model(model)
    try:
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens for model {model}: {str(e)}. Falling back to character-based estimation.")
        return estimate_tokens_by_char_count(text)

def estimate_tokens_by_char_count(text: str) -> int:
    """Estimate token count based on character length."""
    return math.ceil(len(text) / 3)  # Roughly 3 characters per token on average

@retry(wait=wait_exponential(multiplier=1, min=4, max=10),
       stop=stop_after_attempt(3))
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic."""
    return llm.invoke(prompt)

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
    return cleaned.strip('"')

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

def update_state(current_state: AgentState, 
                 content: str, 
                 validation_results: Dict[str, Any], 
                 feedback: List[str], 
                 status: str) -> AgentState:
    """Centralized state update logic."""
    current_attempt = current_state['attempt_count'] + 1
    generation_attempt = {
        "attempt_number": current_attempt,
        "content": content,
        "feedback": " ".join(feedback) if feedback else "Keine Probleme gefunden.",
        "validation_results": validation_results
    }
    return AgentState(
        component=current_state['component'],
        generated_content=content,
        validation_results=validation_results,
        errors=current_state['errors'] + feedback,
        attempt_count=current_attempt,
        status=status,
        generation_history=current_state['generation_history'] + [generation_attempt]
    )

def generate_content(state: AgentState) -> AgentState:
    """Generate copy using LLM based on component requirements."""
    current_attempt = state['attempt_count'] + 1
    logger.info(f"Starting generation attempt #{current_attempt}")
    
    try:
        # Check for max attempts first
        if current_attempt > state['component'].max_attempts:
            return update_state(
                current_state=state,
                content=state['generated_content'],
                validation_results=validate_limits(
                    state['generated_content'],
                    state['component'].char_limit,
                    state['component'].token_limit
                ),
                feedback=[f"Maximum attempts ({state['component'].max_attempts}) reached"],
                status=f"stopped_max_attempts_reached_{state['component'].max_attempts}"
            )

        # Get knowledge base
        kb = get_template_kb()
        # Fetch the briefing
        module_element = state['component'].name  # Assuming 'name' corresponds to 'module_element'
        briefing_text = kb.get_briefing(module_element)
        
        # Fetch the template
        template = kb.get_template(
            state['component'].component_type,
            state['component'].element_type
        )
        
        # Build previous attempts feedback
        previous_attempts_feedback = ""
        if state['generation_history']:
            previous_attempts_feedback = "\nVORHERIGE VERSUCHE UND FEEDBACK:\n"
            for attempt in state['generation_history']:
                previous_attempts_feedback += f"""
Versuch #{attempt['attempt_number']}:
Content: {attempt['content']} ({len(attempt['content'])} Zeichen)
Feedback: {attempt['feedback']}
---"""
        
        # Get examples from knowledge base
        examples = ""
        if template and template.examples:
            examples = "\nBEISPIELE:\n"
            for ex in template.examples[:2]:
                if ex.output:
                    examples += f"Beispiel Output ({len(ex.output)} Zeichen): {ex.output}\n"
        
        # Get specific rules
        specific_rules = ""
        if template and template.rules:
            specific_rules = "\nKOMPONENTEN-SPEZIFISCHE REGELN:\n" + "\n".join(
                f"- {rule}" for rule in template.rules[:5]
            )

        # Determine length guidance based on component type
        length_guidance = """
LÄNGENHINWEISE:
- Nutze kurze, prägnante Wörter
- Vermeide Füllwörter und lange Komposita
- Jedes Wort muss einen Mehrwert bringen
- Nutze Ausrufezeichen sparsam
""" if state['component'].char_limit <= 35 else """
LÄNGENHINWEISE:
- Halte die Sätze kurz und prägnant
- Vermeide Füllwörter und Wiederholungen
- Jeder Satz sollte einen klaren Mehrwert haben
"""

        # Construct prompt
        prompt = f"""Du bist ein Decathlon CRM Copywriter. 
Erstelle motivierenden Content für {state['component'].audience}-Enthusiasten.

KONTEXT UND MARKENIDENTITÄT:
- Freundlich und Einladend: Persönliche, einladende Note
- Begeistert und Positiv: Freude am Sport vermitteln
- Kundenorientiert und Unterstützend: Inspirierende Inhalte
- Einfach und Direkt: Verständliche Sprache
- Spielerisch und Energetisch: Leichter Humor und Dynamik

Briefing: {briefing_text if briefing_text else ''}

{specific_rules}

{examples}

{length_guidance}

WICHTIGE REGELN:
- Keine direkte Anrede
- Keine CTAs oder Verlinkungen im Einführungstext
- STRENGE Längenbegrenzung: Maximal {state['component'].char_limit} Zeichen (inkl. Leerzeichen und Satzzeichen)
- Der Content muss spezifisch auf {state['component'].audience} ausgerichtet sein
- Gib NUR den reinen Text zurück, ohne Präfixe wie 'Text:' oder 'Content:'
- Keine Anführungszeichen verwenden

{previous_attempts_feedback}

Erstelle den Text in deutscher Sprache und beachte dabei das Feedback aus vorherigen Versuchen.
WICHTIG: Der Text MUSS kürzer als {state['component'].char_limit} Zeichen sein. Zähle jeden Buchstaben, jedes Leerzeichen und jedes Satzzeichen."""

        # Add backoff between attempts
        time.sleep(current_attempt * 2)
        
        # Generate content
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.4,
            max_retries=3,
            request_timeout=30
        )
        response = generate_with_retry(llm, prompt)
        
        # Clean and validate the generated content
        new_content = clean_content(response.content)
        validation_results = validate_limits(
            new_content,
            state['component'].char_limit,
            state['component'].token_limit
        )
        
        # Log generation results
        logger.info(f"Generated content with {validation_results['char_count']} characters "
                   f"and {validation_results['token_count']} tokens")
        
        # Generate initial feedback if content exceeds limits
        feedback = []
        if not validation_results['within_char_limit']:
            feedback.append(f"Generated content exceeds character limit: "
                          f"{validation_results['char_count']} > {state['component'].char_limit}")
        
        # Update and return new state
        return update_state(
            current_state=state,
            content=new_content,
            validation_results=validation_results,
            feedback=feedback,
            status="generation_in_progress"
        )
            
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)
        
        # Handle error state using the same update_state function
        return update_state(
            current_state=state,
            content=state['generated_content'],
            validation_results=state['validation_results'] or validate_limits(
                state['generated_content'],
                state['component'].char_limit,
                state['component'].token_limit
            ),
            feedback=[error_msg],
            status="error_during_generation"
        )

def validate_content(state: AgentState) -> AgentState:
    """Validate the generated content against requirements."""
    logger.info("Starting content validation")
    
    # Clean the content
    cleaned_content = clean_content(state['generated_content'])
    
    # Validate limits
    validation_results = validate_limits(
        cleaned_content, 
        state['component'].char_limit, 
        state['component'].token_limit
    )
    
    # Generate feedback
    feedback = generate_feedback(
        validation_results,
        state['component'].char_limit,
        state['component'].token_limit
    )
    
    # Determine status
    new_status = "completed" if validation_results['within_char_limit'] else "validation_failed"
    
    # Update and return new state
    return update_state(
        current_state=state,
        content=cleaned_content,
        validation_results=validation_results,
        feedback=feedback,
        status=new_status
    )

def should_regenerate(state: AgentState) -> str:
    """Determine if content should be regenerated based on validation and attempt count."""
    validation_results = state['validation_results']
    current_attempts = state['attempt_count']
    max_attempts = state['component'].max_attempts
    
    if current_attempts >= max_attempts:
        logger.info(f"Reached maximum attempts ({max_attempts})")
        return END
    
    # Only regenerate if character limit is exceeded or content is empty
    should_regen = not validation_results['within_char_limit'] or validation_results['is_empty']
    logger.info(f"Should regenerate: {should_regen} (Attempt {current_attempts}/{max_attempts})")
    
    if not should_regen:
        return END
    
    if current_attempts < max_attempts:
        return "generate"
    
    return END

def export_results(state: AgentState) -> AgentState:
    """Export the generated content and metadata to a JSON file."""
    logger.info("Exporting results to JSON")
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "component": {
            "name": state['component'].name,
            "type": state['component'].component_type,
            "element": state['component'].element_type,
            "audience": state['component'].audience,
            "token_limit": state['component'].token_limit,
            "char_limit": state['component'].char_limit
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
    filename = f"copy_export_{timestamp}_{state['component'].name.replace(' ', '_')}.json"
    export_path = export_dir / filename
    
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results exported to {export_path}")
    
    return state

class DecathlonCopywriterAgent:
    def __init__(self):
        self.client = Client()
        self.tracer = LangChainTracer(project_name="Decathlon_Agent")
        
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.4,
            callbacks=[self.tracer],
            max_retries=3,
            request_timeout=30
        )
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create the workflow graph with proper state schema."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("export", self._export_node)
        
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "validate")
        
        workflow.add_conditional_edges(
            "validate",
            self._should_regenerate,
            {
                END: "export",
                "generate": "generate"
            }
        )
        
        workflow.add_edge("export", END)
        
        return workflow.compile()
    
    def _generate_node(self, state: AgentState) -> AgentState:
        return generate_content(state)
    
    def _validate_node(self, state: AgentState) -> AgentState:
        return validate_content(state)
    
    def _should_regenerate(self, state: AgentState) -> str:
        return should_regenerate(state)
    
    def _export_node(self, state: AgentState) -> AgentState:
        return export_results(state)
    
    def generate_copy(self, component: CopyComponent) -> AgentState:
        """Public interface for generating copy."""
        logger.info(f"Generating copy for component: {component.name}")
        initial_state = create_initial_state(component)
        return self.workflow.invoke(initial_state)

# Example usage
if __name__ == "__main__":
    agent = DecathlonCopywriterAgent()
    
    test_components = [
        CopyComponent(
            name="headline basic headline",
            token_limit=45,  # Approximately 160 characters
            char_limit=160,
            audience="Schwimmen",
            component_type="headline basic",
            element_type="headline"
        ),
        CopyComponent(
            name="KIT_3_LEFT title",
            token_limit=10,  # Approximately 30-35 characters
            char_limit=25,
            audience="Schwimmen",
            component_type="KIT_3_LEFT",
            element_type="title"
        )
    ]
    
    for component in test_components:
        try:
            print(f"\n{'='*80}")
            print(f"Testing component: {component.name}")
            print(f"Token limit: {component.token_limit}")
            print(f"Character limit: {component.char_limit}")
            print(f"{'='*80}")
            
            result = agent.generate_copy(component)
            
            # Print detailed results
            print("\nGeneration Results:")
            print(f"Status: {result['status']}")
            print(f"Total attempts: {result['attempt_count']}")
            print(f"\nFinal Content:")
            print(f"Length: {len(result['generated_content'])} chars, {result['validation_results']['token_count']} tokens")
            print(f"Content: {result['generated_content']}")
            
            # Print validation details
            print("\nValidation Results:")
            print(f"Character count: {result['validation_results']['char_count']}")
            print(f"Token count: {result['validation_results']['token_count']}")
            print(f"Within token limit: {result['validation_results']['within_token_limit']}")
            print(f"Is empty: {result['validation_results']['is_empty']}")
            
            if result['errors']:
                print("\nErrors encountered:")
                for error in result['errors']:
                    print(f"- {error}")
            
            print("\nGeneration History:")
            for attempt in result['generation_history']:
                print(f"\nAttempt #{attempt['attempt_number']}:")
                print(f"Content: {attempt['content']}")
                print(f"Tokens: {attempt['validation_results']['token_count']}")
                print(f"Feedback: {attempt['feedback']}")
            
            # Add cooldown between components
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error processing component {component.name}: {str(e)}")
            raise