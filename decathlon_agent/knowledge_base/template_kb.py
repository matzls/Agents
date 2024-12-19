"""
Knowledge base classes and loading functions for the Decathlon CRM Copywriter.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from pathlib import Path

@dataclass
class ComponentExample:
    output: str  # Adjusted based on our JSON structure

@dataclass
class ComponentTemplate:
    component_type: str
    element_type: str
    char_limit: int
    examples: List[ComponentExample]
    rules: List[str]

class TemplateKnowledgeBase:
    def __init__(self, templates: Dict[str, ComponentTemplate], briefings: Dict[str, Dict[str, str]]):
        self.templates = templates
        self.briefings = briefings
    
    @classmethod
    def load(cls) -> 'TemplateKnowledgeBase':
        """Load the knowledge base from the JSON files."""
        kb_path = Path(__file__).parent / "decathlon_template_kb.json"
        briefings_path = Path(__file__).parent / "briefings.json"
        
        # Load templates
        with open(kb_path, 'r', encoding='utf-8') as f:
            templates_list = json.load(f)
        
        templates = {}
        for data in templates_list:
            key = f"{data['component_type']}_{data['element_type']}".lower()
            templates[key] = ComponentTemplate(
                component_type=data["component_type"],
                element_type=data["element_type"],
                char_limit=data["char_limit"],
                examples=[ComponentExample(**ex) for ex in data.get("examples", [])],
                rules=data.get("rules", [])
            )
        
        # Load briefings
        with open(briefings_path, 'r', encoding='utf-8') as f:
            briefings_dict = json.load(f)
        
        briefings = briefings_dict  # Keys are module_elements
        
        return cls(templates=templates, briefings=briefings)
    
    def get_template(self, component_type: str, element_type: str) -> Optional[ComponentTemplate]:
        """Get a template by component and element type."""
        key = f"{component_type}_{element_type}".lower()
        return self.templates.get(key)
    
    def get_briefing(self, module_element: str) -> Optional[Dict[str, str]]:
        """Get the briefing data for a given module element."""
        return self.briefings.get(module_element.lower())
    
    def get_examples_for_component(self, component_type: str, element_type: str, 
                                   max_examples: int = 2) -> List[ComponentExample]:
        """Get relevant examples for a component type."""
        template = self.get_template(component_type, element_type)
        if template:
            return template.examples[:max_examples]
        return []
    
    def get_rules_for_component(self, component_type: str, element_type: str) -> List[str]:
        """Get all rules for a component type."""
        template = self.get_template(component_type, element_type)
        if template:
            return template.rules
        return []

# Create a singleton instance
template_kb = None

def get_template_kb() -> TemplateKnowledgeBase:
    """Get or create the template knowledge base singleton."""
    global template_kb
    if template_kb is None:
        template_kb = TemplateKnowledgeBase.load()
    return template_kb