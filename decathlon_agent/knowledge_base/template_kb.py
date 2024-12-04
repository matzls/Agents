"""
Knowledge base classes and loading functions for the Decathlon CRM Copywriter.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import json
from pathlib import Path

@dataclass
class ComponentExample:
    briefing: str
    output: str
    context: str

@dataclass
class ComponentTemplate:
    component_type: str
    element_type: str
    char_limit: int
    examples: List[ComponentExample]
    rules: List[str]

class TemplateKnowledgeBase:
    def __init__(self, templates: Dict[str, ComponentTemplate]):
        self.templates = templates
    
    @classmethod
    def load(cls) -> 'TemplateKnowledgeBase':
        """Load the knowledge base from the JSON file."""
        kb_path = Path(__file__).parent / "decathlon_template_kb.json"
        
        with open(kb_path, 'r', encoding='utf-8') as f:
            kb_dict = json.load(f)
        
        templates = {
            name: ComponentTemplate(
                component_type=data["component_type"],
                element_type=data["element_type"],
                char_limit=data["char_limit"],
                examples=[ComponentExample(**ex) for ex in data["examples"]],
                rules=data["rules"]
            )
            for name, data in kb_dict.items()
        }
        
        return cls(templates=templates)
    
    def get_template(self, component_type: str, element_type: str) -> Optional[ComponentTemplate]:
        """Get a template by component and element type."""
        key = f"{component_type}_{element_type}".lower()
        return self.templates.get(key)
    
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