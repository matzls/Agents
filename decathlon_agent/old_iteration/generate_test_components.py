"""
Script to generate test_components based on the knowledge base and briefing file.
"""

from pathlib import Path
import json
from dataclasses import dataclass, asdict
from typing import Optional, List

@dataclass(frozen=True)
class CopyComponent:
    name: str  # Should correspond to 'module_element' in briefings
    token_limit: int
    char_limit: int
    briefing: str
    audience: str
    component_type: str
    element_type: str
    max_attempts: int = 3
    url: Optional[str] = None

def generate_test_components():
    # Define paths
    current_dir = Path(__file__).parent
    knowledge_base_path = current_dir / "knowledge_base" / "decathlon_template_kb.json"
    briefings_path = current_dir / "knowledge_base" / "briefings.json"
    
    # Output path
    test_components_path = current_dir / "test_components.json"

    # Load knowledge base and briefings
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)

    with open(briefings_path, 'r', encoding='utf-8') as f:
        briefings = json.load(f)

    test_components = []

    for template in templates:
        component_type = template['component_type'].strip()
        element_type = template['element_type'].strip()
        char_limit = int(template['char_limit'])

        # Construct module_element key to match briefing entries
        module_element = f"{component_type} {element_type}".lower()

        # Attempt to find a matching briefing
        briefing_data = briefings.get(module_element)

        if not briefing_data:
            # If no briefing is found, skip or handle as needed
            print(f"No briefing found for module element: {module_element}")
            continue

        copy_briefing = briefing_data['copy_briefing']
        audience = briefing_data['audience']

        # Estimate token_limit based on char_limit
        token_limit = max(1, char_limit // 4)  # Adjust estimation as needed

        component = CopyComponent(
            name=module_element,
            token_limit=token_limit,
            char_limit=char_limit,
            briefing=copy_briefing,
            audience=audience,
            component_type=component_type,
            element_type=element_type
        )

        test_components.append(component)

    # Save test_components to JSON
    with open(test_components_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(comp) for comp in test_components], f, ensure_ascii=False, indent=2)

    print(f"Generated {len(test_components)} test components.")
    print(f"Test components saved to: {test_components_path}")

    return test_components

if __name__ == "__main__":
    generate_test_components()