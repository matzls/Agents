"""
Decathlon CRM Copywriter POC using LangGraph.
This script implements a graph-based workflow for generating and validating marketing copy.
"""

import os
import time
import operator
import json
from datetime import datetime
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain.callbacks.tracers import LangChainTracer
from langsmith import Client
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

# Add knowledge base import
from knowledge_base.template_kb import get_template_kb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass(frozen=True)
class CopyComponent:
    name: str
    char_limit: int
    briefing: str
    audience: str
    component_type: str
    element_type: str
    max_attempts: int = 3
    url: Optional[str] = None

@dataclass(frozen=True)
class ValidationResult:
    char_count: int
    within_limit: bool
    is_empty: bool

@dataclass(frozen=True)
class GenerationAttempt:
    attempt_number: int
    content: str
    feedback: str
    validation_results: ValidationResult

class AgentState(TypedDict):
    """Main state class for the copywriting workflow."""
    component: CopyComponent
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[GenerationAttempt]

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

class OpenAIError(Exception):
    """Custom exception for OpenAI API errors with detailed information."""
    def __init__(self, status_code, error_message, request_id=None):
        self.status_code = status_code
        self.error_message = error_message
        self.request_id = request_id
        super().__init__(self.get_detailed_message())

    def get_detailed_message(self):
        message = f"OpenAI API Error (Status {self.status_code}): {self.error_message}"
        if self.request_id:
            message += f"\nRequest ID: {self.request_id}"
        return message

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    reraise=True
)
def generate_with_retry(llm, prompt: str):
    """Generate content with retry logic and detailed error handling."""
    try:
        logger.info("Making API request to OpenAI")
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        error_info = str(e)
        status_code = None
        request_id = None
        
        # Extract error details from different exception types
        if hasattr(e, 'response'):
            try:
                error_data = e.response.json()
                error_info = error_data.get('error', {}).get('message', str(e))
                status_code = e.response.status_code
                request_id = e.response.headers.get('x-request-id')
            except:
                pass
        elif hasattr(e, 'status_code'):
            status_code = e.status_code
        
        error_details = {
            "status_code": status_code or 500,
            "error_message": error_info,
            "request_id": request_id
        }
        
        logger.error(
            "OpenAI API Error Details:\n"
            f"Status Code: {error_details['status_code']}\n"
            f"Error Message: {error_details['error_message']}\n"
            f"Request ID: {error_details['request_id']}\n"
            "Response Headers: %s",
            getattr(getattr(e, 'response', None), 'headers', 'No headers available')
        )
        
        # On 500 error, add retry information
        if status_code == 500:
            logger.warning("Encountered 500 error, will retry with exponential backoff")
        
        raise OpenAIError(
            status_code=error_details['status_code'],
            error_message=error_details['error_message'],
            request_id=error_details['request_id']
        )

def generate_content(state: AgentState) -> AgentState:
    """Generate copy using LLM based on component requirements."""
    current_attempt = state['attempt_count'] + 1
    logger.info(f"Starting generation attempt #{current_attempt}")
    
    # Get knowledge base
    kb = get_template_kb()
    
    if current_attempt > state['component'].max_attempts:
        return AgentState(
            component=state['component'],
            generated_content=state['generated_content'],
            validation_results=state['validation_results'],
            errors=state['errors'],
            attempt_count=current_attempt,
            status=f"stopped_max_attempts_reached_{state['component'].max_attempts}",
            generation_history=state['generation_history']
        )

    try:
        # Get template information
        template = kb.get_template(
            state['component'].component_type,
            state['component'].element_type
        )
        
        previous_attempts_feedback = ""
        if state['generation_history']:
            previous_attempts_feedback = "\nVORHERIGE VERSUCHE UND FEEDBACK:\n"
            for attempt in state['generation_history']:
                previous_attempts_feedback += f"""
Versuch #{attempt.attempt_number}:
Content: {attempt.content}
Feedback: {attempt.feedback}
---"""
        
        # Get examples from knowledge base
        examples = ""
        if template:
            template_examples = template.examples[:2]  # Get up to 2 examples
            if template_examples:
                examples = "\nBEISPIELE:\n"
                for ex in template_examples:
                    if ex.output:
                        examples += f"Beispiel Output: {ex.output}\n"
        
        # Get specific rules for this component type
        specific_rules = ""
        if template:
            rules = template.rules
            if rules:
                specific_rules = "\nKOMPONENTEN-SPEZIFISCHE REGELN:\n" + "\n".join(f"- {rule}" for rule in rules[:5])

        prompt = f"""Du bist ein Decathlon CRM Copywriter. 
Erstelle motivierenden Content für {state['component'].audience}-Enthusiasten.

KONTEXT UND MARKENIDENTITÄT:
- Freundlich und Einladend: Persönliche, einladende Note
- Begeistert und Positiv: Freude am Sport vermitteln
- Kundenorientiert und Unterstützend: Inspirierende Inhalte
- Einfach und Direkt: Verständliche Sprache
- Spielerisch und Energetisch: Leichter Humor und Dynamik

{specific_rules}

{examples}

Briefing: {state['component'].briefing}

WICHTIGE REGELN:
- Keine direkte Anrede
- Keine CTAs oder Verlinkungen im Einführungstext
- Maximale Zeichenlänge: {state['component'].char_limit} Zeichen
- Der Content muss spezifisch auf {state['component'].audience} ausgerichtet sein
- Gib NUR den reinen Text zurück, ohne Präfixe wie 'Text:' oder 'Content:'
- Keine Anführungszeichen verwenden

{previous_attempts_feedback}

Erstelle den Text in deutscher Sprache und beachte dabei das Feedback aus vorherigen Versuchen.
WICHTIG: Der Text darf maximal {state['component'].char_limit} Zeichen lang sein."""

        # Add backoff between attempts
        time.sleep(state['attempt_count'] * 2)
        
        try:
            response = generate_with_retry(ChatOpenAI(
                model="gpt-4",
                temperature=0.4,
                max_retries=3,
                request_timeout=30
            ), prompt)
            new_content = response.content
            logger.info(f"Generated content with {len(new_content)} characters")
            print(f"\nGenerated content:")
            print(new_content)
            
        except OpenAIError as e:
            logger.error(f"OpenAI API Error during attempt #{current_attempt}:\n{str(e)}")
            if current_attempt >= state['component'].max_attempts:
                return AgentState(
                    component=state['component'],
                    generated_content="",
                    validation_results=state['validation_results'],
                    errors=state['errors'] + [f"Final attempt failed: {str(e)}"],
                    attempt_count=current_attempt,
                    status="error_max_attempts_with_api_failure",
                    generation_history=state['generation_history']
                )
            raise  # Re-raise to trigger retry
        
        return AgentState(
            component=state['component'],
            generated_content=new_content,
            validation_results=state['validation_results'],
            errors=state['errors'],
            attempt_count=current_attempt,
            status="generation_in_progress",
            generation_history=state['generation_history']
        )
        
    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        if isinstance(e, OpenAIError):
            error_msg = str(e)  # Use the detailed error message
        logger.error(error_msg)
        return AgentState(
            component=state['component'],
            generated_content=state['generated_content'],
            validation_results=state['validation_results'],
            errors=state['errors'] + [error_msg],
            attempt_count=current_attempt,
            status="error_during_generation",
            generation_history=state['generation_history']
        )
    
def validate_content(state: AgentState) -> AgentState:
    """Validate the generated content against requirements."""
    logger.info("Starting content validation")
    content = state['generated_content']
    char_limit = state['component'].char_limit
    current_attempt = state['attempt_count']
    
    # Clean the content by removing any prefixes and quotes
    cleaned_content = content.strip()
    # Remove common prefixes
    prefixes = ['Text:', 'Content:', 'Text: ', 'Content: ']
    for prefix in prefixes:
        if cleaned_content.lower().startswith(prefix.lower()):
            cleaned_content = cleaned_content[len(prefix):].strip()
    # Remove quotes
    cleaned_content = cleaned_content.strip('"\'')
    
    validation = ValidationResult(
        char_count=len(cleaned_content),
        within_limit=len(cleaned_content) <= char_limit,
        is_empty=not cleaned_content.strip()
    )
    
    logger.info(f"Character count: {validation.char_count}/{char_limit}")
    
    feedback = []
    if not validation.within_limit:
        feedback.append(f"Der Text ist zu lang ({len(cleaned_content)} Zeichen). Bitte kürze auf maximal {char_limit} Zeichen.")
    if validation.is_empty:
        feedback.append("Der generierte Text ist leer. Bitte erstelle einen neuen Text.")
    
    attempt = GenerationAttempt(
        attempt_number=current_attempt,
        content=cleaned_content,  # Store the cleaned content
        feedback=" ".join(feedback) if feedback else "Keine Probleme gefunden.",
        validation_results=validation
    )
    
    # Update status based on validation results
    new_status = "completed" if validation.within_limit else "validation_failed"
    
    return AgentState(
        component=state['component'],
        generated_content=cleaned_content,  # Use cleaned content in state
        validation_results=validation.__dict__,
        errors=state['errors'] + feedback,
        attempt_count=current_attempt,
        status=new_status,
        generation_history=state['generation_history'] + [attempt]
    )

def should_regenerate(state: AgentState) -> str:
    """Determine if content should be regenerated based on validation and attempt count."""
    validation = ValidationResult(**state['validation_results'])
    max_attempts = state['component'].max_attempts
    current_attempts = state['attempt_count']
    
    # Check for errors
    if state['status'] == "error_during_generation":
        logger.info("Stopping due to generation error")
        return END
    
    # Check max attempts
    if current_attempts >= max_attempts:
        logger.info(f"Reached maximum attempts ({max_attempts})")
        return END
    
    should_regen = not validation.within_limit or validation.is_empty
    logger.info(f"Should regenerate: {should_regen} (Attempt {current_attempts}/{max_attempts})")
    
    if not should_regen:
        return END
    
    return "generate"

def export_results(state: AgentState) -> AgentState:
    """Export the generated content and metadata to a JSON file."""
    logger.info("Exporting results to JSON")
    
    # Create export data structure
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "component": {
            "name": state['component'].name,
            "type": state['component'].component_type,
            "element": state['component'].element_type,
            "audience": state['component'].audience,
            "char_limit": state['component'].char_limit,
            "briefing": state['component'].briefing
        },
        "generation_result": {
            "final_content": state['generated_content'],
            "char_count": len(state['generated_content']),
            "status": state['status'],
            "total_attempts": state['attempt_count']
        },
        "generation_history": [
            {
                "attempt": attempt.attempt_number,
                "content": attempt.content,
                "feedback": attempt.feedback,
                "validation": {
                    "char_count": attempt.validation_results.char_count,
                    "within_limit": attempt.validation_results.within_limit,
                    "is_empty": attempt.validation_results.is_empty
                }
            }
            for attempt in state['generation_history']
        ]
    }
    
    # Create exports directory if it doesn't exist
    export_dir = Path("exports")
    export_dir.mkdir(exist_ok=True)
    
    # Create filename with timestamp and component info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"copy_export_{timestamp}_{state['component'].name}.json"
    export_path = export_dir / filename
    
    # Save to JSON file
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results exported to {export_path}")
    
    return state

class DecathlonCopywriterAgent:
    def __init__(self):
        self.client = Client()
        self.tracer = LangChainTracer(project_name="Decathlon_Agent")
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            callbacks=[self.tracer],
            max_retries=3,
            request_timeout=30
        )
        
        self.workflow = self._create_workflow()
    
    def _create_workflow(self):
        """Create the workflow graph with proper state schema."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("export", self._export_node)  # Add export node
        
        # Add edges
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", "validate")
        
        # Add conditional edges from validate
        workflow.add_conditional_edges(
            "validate",
            self._should_regenerate,
            {
                END: "export",  # Go to export when validation is complete
                "generate": "generate"  # Go back to generate if needed
            }
        )
        
        # Add final edge from export to END
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
            name="swimming_title",
            char_limit=30,
            briefing="Ab ins Wasser - kurze, knackige Motivation",
            audience="Schwimmen",
            component_type="headline",
            element_type="title"
        ),
        CopyComponent(
            name="swimming_copy",
            char_limit=160,
            briefing="Ab ins Wasser und richtig auspowern - motiviere die Schwimmer zum Jahresendspurt",
            audience="Schwimmen",
            component_type="headline",
            element_type="copy"
        )
    ]
    
    for component in test_components:
        
            print(f"\n{'='*80}")
            print(f"Testing component: {component.name}")
            print(f"Character limit: {component.char_limit}")
            print(f"{'='*80}")
            
            result = agent.generate_copy(component)
            
            # Print detailed results
            print("\nGeneration Results:")
            print(f"Status: {result['status']}")
            print(f"Total attempts: {result['attempt_count']}")
            print(f"\nFinal Content:")
            print(f"Length: {len(result['generated_content'])} chars")
            print(f"Content: {result['generated_content']}")
            
            # Print validation details
            print("\nValidation Results:")
            print(f"Character count: {result['validation_results']['char_count']}")
            print(f"Within limit: {result['validation_results']['within_limit']}")
            print(f"Is empty: {result['validation_results']['is_empty']}")
            
            if result['errors']:
                print("\nErrors encountered:")
                for error in result['errors']:
                    print(f"- {error}")
            
            print("\nGeneration History:")
            for attempt in result['generation_history']:
                print(f"\nAttempt #{attempt.attempt_number}:")
                print(f"Content: {attempt.content}")
                print(f"Feedback: {attempt.feedback}")