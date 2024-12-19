import json
from typing import List, Dict, Any, Optional

class TemplateKnowledgeBase:
    def __init__(self, kb_data: List[Dict[str, Any]]):
        self.kb_data = kb_data

    def get_template(self, component_type: str, element_type: str) -> Optional[Dict[str, Any]]:
        for component in self.kb_data:
            if component['component_type'] == component_type and component['element_type'] == element_type:
                return component
        return None

def get_template_kb() -> TemplateKnowledgeBase:
    with open('decathlon_template_kb.json', 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    return TemplateKnowledgeBase(kb_data)