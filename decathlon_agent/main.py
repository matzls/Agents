import os
from typing import Dict, List, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DecathlonEmailState:
    """
    State management class for Decathlon email generation workflow
    """
    def __init__(self):
        self.briefing: Dict[str, Any] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.url_contents: List[str] = []
        self.generated_variants: Dict[str, List[str]] = {
            'subject_lines': [],
            'preheaders': [],
            'headlines': [],
            'body_copies': []
        }
        self.validation_results: Dict[str, bool] = {}
        self.rejection_info: Dict[str, Any] = {
            'subject_lines': [],
            'preheaders': [],
            'headlines': [],
            'body_copies': []
        }
        self.attempt_counts: Dict[str, int] = {
            'subject_lines': 0,
            'preheaders': 0,
            'headlines': 0,
            'body_copies': 0
        }
        self.max_retries: int = 3

def load_knowledge_base(guidelines_path: str) -> Dict[str, Any]:
    """
    Load brand guidelines from markdown file
    
    Args:
        guidelines_path (str): Path to guidelines markdown file
    
    Returns:
        Dict[str, Any]: Parsed guidelines
    """
    try:
        with open(guidelines_path, 'r') as f:
            guidelines_content = f.read()
        
        # Basic parsing - can be enhanced with more sophisticated markdown parsing
        knowledge_base = {
            'character_limits': {},
            'tone_specifications': {},
            'structural_requirements': {},
            'brand_voice': {}
        }
        
        # TODO: Implement more robust parsing logic
        logger.info("Knowledge base loaded successfully")
        return knowledge_base
    
    except FileNotFoundError:
        logger.error(f"Guidelines file not found at {guidelines_path}")
        return {}

def create_workflow(llm: BaseLanguageModel):
    """
    Create LangGraph workflow for email generation
    
    Args:
        llm (BaseLanguageModel): Language model for content generation
    
    Returns:
        StateGraph: Configured workflow
    """
    def web_crawling_node(state: DecathlonEmailState):
        # TODO: Implement web content extraction
        logger.info("Performing web crawling")
        return state
    
    def content_generation_node(state: DecathlonEmailState):
        logger.info("Generating content variants")
        
        components = ['subject_lines', 'preheaders', 'headlines', 'body_copies']
        
        for component in components:
            # Skip if max retries reached
            if state.attempt_counts[component] >= state.max_retries:
                logger.warning(f"Max retries reached for {component}")
                continue
                
            # Get rejection info if available
            rejection_context = ""
            if state.rejection_info[component]:
                last_rejection = state.rejection_info[component][-1]
                rejection_context = f"\nPrevious generation was rejected because: {last_rejection['reason']}. "
                rejection_context += f"Previous content: {last_rejection['content']}\n"
                rejection_context += "Please address these issues in the new generation."
            
            # Generate content with rejection context
            try:
                prompt = f"""Generate email content for {component}.
                Knowledge base context: {state.knowledge_base}
                Briefing: {state.briefing}
                {rejection_context}
                """
                
                response = llm.invoke(prompt)
                state.generated_variants[component].append(response)
                state.attempt_counts[component] += 1
                
            except Exception as e:
                logger.error(f"Error generating {component}: {str(e)}")
        
        return state
    
    def validation_node(state: DecathlonEmailState):
        logger.info("Validating generated content")
        
        components = ['subject_lines', 'preheaders', 'headlines', 'body_copies']
        
        for component in components:
            if not state.generated_variants[component]:
                continue
                
            latest_content = state.generated_variants[component][-1]
            
            # Validate content
            try:
                validation_prompt = f"""Validate this {component} content:
                Content: {latest_content}
                Requirements: {state.knowledge_base}
                
                Return JSON with:
                - binary_score: 'yes' or 'no'
                - reason: explanation if rejected
                """
                
                result = llm.invoke(validation_prompt)
                validation_result = result.json()
                
                if validation_result['binary_score'] == 'no':
                    # Store rejection info
                    rejection_info = {
                        'reason': validation_result['reason'],
                        'content': latest_content,
                        'attempt': state.attempt_counts[component]
                    }
                    state.rejection_info[component].append(rejection_info)
                    
                    # Check if should regenerate
                    if state.attempt_counts[component] < state.max_retries:
                        logger.info(f"Content rejected for {component}, will regenerate")
                        state.validation_results[component] = False
                    else:
                        logger.warning(f"Max retries reached for {component}")
                        state.validation_results[component] = False
                else:
                    state.validation_results[component] = True
                    # Clear rejection info on success
                    state.rejection_info[component] = []
                    
            except Exception as e:
                logger.error(f"Error validating {component}: {str(e)}")
                state.validation_results[component] = False
        
        # Determine if any components need regeneration
        needs_regeneration = any(
            not result and state.attempt_counts[comp] < state.max_retries
            for comp, result in state.validation_results.items()
        )
        
        if needs_regeneration:
            return "content_generation"
        
        return END
    
    # Initialize workflow
    workflow = StateGraph(DecathlonEmailState)
    
    # Define workflow nodes
    workflow.add_node("web_crawling", web_crawling_node)
    workflow.add_node("content_generation", content_generation_node)
    workflow.add_node("validation", validation_node)
    
    # Define edges
    workflow.set_entry_point("web_crawling")
    workflow.add_edge("web_crawling", "content_generation")
    workflow.add_edge("content_generation", "validation")
    workflow.add_edge("validation", "content_generation")  # Add regeneration edge
    workflow.add_edge("validation", END)
    
    return workflow.compile()

def main():
    # Load environment variables or use secure configuration
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Initialize language model
    llm = ChatOpenAI(
        api_key=openai_api_key, 
        model="gpt-4-turbo",
        temperature=0.7
    )
    
    # Load knowledge base into 

    guidelines_path = "/Users/mg/Desktop/GitHub/Playground/Agents/decathlon_agent/knowledge_baseDECATHLON_CRM-Copywriter_Instructions.md"
    knowledge_base = load_knowledge_base(guidelines_path)
    
    # Create workflow
    workflow = create_workflow(llm)
    
    # Example usage (to be expanded)
    initial_state = DecathlonEmailState()
    initial_state.knowledge_base = knowledge_base
    
    # Execute workflow
    final_state = workflow.invoke(initial_state)
    
    logger.info("Workflow completed successfully")

if __name__ == "__main__":
    main()
