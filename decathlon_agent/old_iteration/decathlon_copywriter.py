from typing import TypedDict, List, Dict, Any
from datetime import datetime
import logging
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import time
import json
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class CopyComponent(TypedDict):
    """Structure for copy component configuration."""
    component_type: str
    element_type: str
    char_limit: int
    token_limit: int
    audience: str
    name: str

class GenerationAttempt(TypedDict):
    """Structure for tracking generation attempts."""
    attempt_number: int
    content: str
    feedback: List[str]
    timestamp: str

class AgentState(TypedDict):
    """Main state class for the copywriting workflow."""
    component: CopyComponent
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[GenerationAttempt]

def get_template_kb():
    """Placeholder for template knowledge base retrieval."""
    # Implementation needed
    return None

def clean_content(content: str) -> str:
    """Clean and format the generated content."""
    return content.strip()

def validate_limits(content: str, char_limit: int, token_limit: int) -> Dict[str, Any]:
    """Validate content against character and token limits."""
    char_count = len(content)
    # Simple token count approximation
    token_count = len(content.split())
    
    return {
        'char_count': char_count,
        'token_count': token_count,
        'within_char_limit': char_count <= char_limit,
        'within_token_limit': token_count <= token_limit,
        'is_empty': len(content.strip()) == 0
    }

def generate_content(state: AgentState) -> AgentState:
    """Generate copy using LLM based on component requirements."""
    current_attempt = state['attempt_count'] + 1
    logger.info(f"Starting generation attempt #{current_attempt}")

    try:
        # Existing code to build prompt and generate content...

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

        # Return new state directly
        return {
            **state,
            'generated_content': new_content,
            'validation_results': validation_results,
            'attempt_count': current_attempt,
            'generation_history': [
                *state['generation_history'],
                {
                    'attempt_number': current_attempt,
                    'content': new_content,
                    'validation_results': validation_results,
                    'feedback': [],  # Feedback will be added in validation
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'status': "generated"
        }

    except Exception as e:
        error_msg = f"Content generation error: {str(e)}"
        logger.error(error_msg)

        # Ensure validation_results is included even on error
        validation_results = state.get('validation_results', {})
        # Optionally, you can set default values for validation_results here

        # Update generation history
        return {
            **state,
            'errors': [*state['errors'], error_msg],
            'attempt_count': current_attempt,
            'generation_history': [
                *state['generation_history'],
                {
                    'attempt_number': current_attempt,
                    'content': state['generated_content'],
                    'validation_results': validation_results,
                    'feedback': [error_msg],
                    'timestamp': datetime.now().isoformat()
                }
            ],
            'status': "error_during_generation"
        }
    
def validate_content(state: AgentState) -> AgentState:
    """Validate generated content and provide feedback."""
    logger.info("Starting content validation")
    
    validation_results = state['validation_results']
    feedback = []
    
    logger.info("Content validation:")
    logger.info(f"Character count: {validation_results['char_count']}/{state['component']['char_limit']}")
    logger.info(f"Token count: {validation_results['token_count']}/{state['component']['token_limit']}")
    
    if validation_results['is_empty']:
        feedback.append("Generated content is empty")
    if not validation_results['within_char_limit']:
        feedback.append(f"Content exceeds character limit: {validation_results['char_count']}/{state['component']['char_limit']}")
    if not validation_results['within_token_limit']:
        feedback.append(f"Content exceeds token limit: {validation_results['token_count']}/{state['component']['token_limit']}")
    
    # Update the last generation attempt with feedback
    updated_history = state['generation_history'].copy()
    if updated_history:
        updated_history[-1] = {
            **updated_history[-1],
            'feedback': feedback
        }
    
    return {
        **state,
        'validation_results': validation_results,
        'errors': feedback,
        'generation_history': updated_history,
        'status': "completed" if not feedback else "needs_regeneration"
    }

def initialize_state(component: CopyComponent) -> AgentState:
    """Initialize agent state for a component."""
    return {
        'component': component,
        'generated_content': "",
        'validation_results': {},
        'errors': [],
        'attempt_count': 0,
        'status': "initialized",
        'generation_history': []
    }
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



### testing
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
                    if 'feedback' in attempt and attempt['feedback']:
                        print(f"Feedback: {', '.join(attempt['feedback'])}")

            time.sleep(5)  # Add cooldown between components

        except Exception as e:
            logger.error(f"Error processing component {component.component_type} - {component.element_type}: {str(e)}")
            continue