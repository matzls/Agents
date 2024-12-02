"""
Decathlon CRM Copywriter POC using LangGraph.
This script implements a graph-based workflow for generating and validating marketing copy.
"""

import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from langsmith import Client

# Load environment variables
load_dotenv()

# Type definitions
class CopyComponent:
    def __init__(
        self,
        name: str,
        char_limit: int,
        briefing: str,
        audience: str,
        max_attempts: int = 3
    ):
        self.name = name
        self.char_limit = char_limit
        self.briefing = briefing
        self.audience = audience
        self.max_attempts = max_attempts

    def to_dict(self):
        return {
            "name": self.name,
            "char_limit": self.char_limit,
            "briefing": self.briefing,
            "audience": self.audience,
            "max_attempts": self.max_attempts
        }

class CopyState(TypedDict):
    component: Dict[str, Any]  # Changed to Dict since we're converting CopyComponent to dict
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str

# Initialize LangSmith client and tracer
client = Client()
tracer = LangChainTracer(
    project_name="Decathlon_Agent"
)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.4,
    callbacks=[tracer],
    callback_manager=CallbackManager([tracer])
)

def initialize_state(component: Dict[str, Any]) -> CopyState:
    """Initialize the workflow state with a component."""
    return {
        "component": component,
        "generated_content": "",
        "validation_results": {},
        "errors": [],
        "attempt_count": 0,
        "status": "initialized"
    }

def generate_content(state: CopyState) -> Dict[str, Any]:
    """Generate copy using LLM based on component requirements."""
    updated_state = state.copy()
    updated_state["attempt_count"] = state.get("attempt_count", 0) + 1
    updated_state["errors"] = []
    
    try:
        print(f"\n=== Generation Attempt #{updated_state['attempt_count']}/{updated_state['component']['max_attempts']} ===")
        prompt = f"""Du bist ein Decathlon CRM Copywriter. 
Erstelle motivierenden Content für {updated_state['component']['audience']}-Enthusiasten.

KONTEXT UND MARKENIDENTITÄT:
- Freundlich und Einladend: Persönliche, einladende Note
- Begeistert und Positiv: Freude am Sport vermitteln
- Kundenorientiert und Unterstützend: Inspirierende Inhalte, klare Orientierung
- Einfach und Direkt: Verständliche Sprache, kein Fachjargon
- Spielerisch und Energetisch: Leichter Humor und Dynamik

Briefing: {updated_state['component']['briefing']}

WICHTIGE REGELN:
- Keine direkte Anrede
- Keine CTAs oder Verlinkungen im Einführungstext
- Exakte Zeichenlänge: {updated_state['component']['char_limit']} Zeichen
- Der Content muss spezifisch auf {updated_state['component']['audience']} ausgerichtet sein
- Nutze einen dynamischen, motivierenden Ton
- Erwähne das Equipment im Kontext von {updated_state['component']['audience']}

Erstelle den Text in deutscher Sprache."""
        
        response = llm.invoke(prompt)
        updated_state["generated_content"] = response.content
        print(f"Generated content ({len(updated_state['generated_content'])} chars):")
        print(updated_state["generated_content"])
        
    except Exception as e:
        updated_state["errors"].append(f"Content generation error: {str(e)}")
    
    return updated_state

def validate_content(state: CopyState) -> Dict[str, Any]:
    """Validate the generated content against requirements."""
    print("\n=== Validation ===")
    updated_state = state.copy()
    content = updated_state["generated_content"]
    char_limit = updated_state["component"]["char_limit"]
    
    validation = {
        "char_count": len(content),
        "within_limit": len(content) <= char_limit,
        "is_empty": not content.strip()
    }
    
    print(f"Character count: {validation['char_count']}/{char_limit}")
    print(f"Within limit: {validation['within_limit']}")
    
    updated_state["validation_results"] = validation
    
    if not validation["within_limit"]:
        error_msg = f"Content exceeds character limit: {len(content)}/{char_limit}"
        updated_state["errors"].append(error_msg)
        print(f"Error: {error_msg}")
    if validation["is_empty"]:
        error_msg = "Generated content is empty"
        updated_state["errors"].append(error_msg)
        print(f"Error: {error_msg}")
    
    return updated_state

def should_regenerate(state: CopyState) -> str:
    """Determine if content should be regenerated based on validation and attempt count."""
    validation = state["validation_results"]
    max_attempts = state["component"]["max_attempts"]
    current_attempts = state["attempt_count"]
    
    if current_attempts >= max_attempts:
        print(f"\nReached maximum attempts ({max_attempts}). Stopping regeneration.")
        state["status"] = f"stopped_max_attempts_reached_{max_attempts}"
        state["errors"].append(f"Maximum attempts ({max_attempts}) reached without successful generation")
        return END
    
    should_regen = not validation["within_limit"] or validation["is_empty"]
    print(f"\nShould regenerate: {should_regen} (Attempt {current_attempts}/{max_attempts})")
    
    if not should_regen:
        state["status"] = "completed_successfully"
        return END
    
    return "generate"

def create_workflow():
    """Create the workflow graph."""
    workflow = StateGraph(CopyState)
    
    # Add nodes
    workflow.add_node("generate", generate_content)
    workflow.add_node("validate", validate_content)
    
    # Add edges
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", "validate")
    
    # Add conditional edge for regeneration
    workflow.add_conditional_edges(
        "validate",
        should_regenerate
    )
    
    return workflow.compile()




def main():
    # Test components
    test_components = [
        CopyComponent(
            name="headline basic",
            char_limit=400,
            briefing="Das Jahresende naht, heißt aber nicht, dass du auch das Jahr ruhig ausklingen lassen musst. Zieh jetzt nochmal richtig an und power dich zum Ende des Jahres nochmal komplett aus und erreiche neue Limits",
            audience="Schwimmen"
        ),
        CopyComponent(
            name="Product Teaser",
            char_limit=50,
            briefing="Neue Winterjacken-Kollektion mit innovativer Thermotechnologie",
            audience="Basketball"
        ),
        CopyComponent(
            name="Season Opening",
            char_limit=200,
            briefing="Start in die Wintersaison mit neuer Ski- und Snowboard",
            audience="Ski"
        )
    ]
    
    # Process each test component
    for i, component in enumerate(test_components, 1):
        print(f"\n\n{'='*50}")
        print(f"TEST CASE {i}: {component.name}")
        print(f"Character Limit: {component.char_limit}")
        print(f"Max Attempts: {component.max_attempts}")
        print(f"Briefing: {component.briefing}")
        print(f"Audience: {component.audience}")
        print(f"{'='*50}")
        
        try:
            initial_state = initialize_state(component.to_dict())
            workflow = create_workflow()
            final_state = workflow.invoke(initial_state)
            
            print("\n=== Final Results ===")
            print(f"Status: {final_state['status']}")
            print(f"Total generation attempts: {final_state['attempt_count']}")
            print(f"\nFinal Content ({len(final_state['generated_content'])} chars):")
            print(final_state['generated_content'])
            print("\nValidation Results:")
            for key, value in final_state['validation_results'].items():
                print(f"{key}: {value}")
            
            if final_state['errors']:
                print("\nErrors encountered during process:")
                for error in final_state['errors']:
                    print(f"- {error}")
            
            print("\n=== Complete State Dictionary ===")
            import json
            print(json.dumps(final_state, indent=2))
                    
        except Exception as e:
            print(f"Workflow execution error: {str(e)}")

if __name__ == "__main__":
    main()