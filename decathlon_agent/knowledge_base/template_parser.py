"""
Tool to parse Excel templates into JSON knowledge base.
Run this script when the template files change.
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any, List

def parse_email_components_excel(excel_path: str) -> List[Dict[str, Any]]:
    """Parse the Email Components Excel file into a list of templates."""
    df = pd.read_excel(excel_path)
    
    templates = []
    
    for _, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(row['Component']) or pd.isna(row['Text element']):
            continue
        
        # Extract data
        component_type = str(row['Component']).strip()
        element_type = str(row['Text element']).strip()
        char_limit = int(row['Character limitation']) if not pd.isna(row['Character limitation']) else 0
        general_topic = str(row['General topic']).strip() if not pd.isna(row['General topic']) else ""
        specific_briefing = str(row['Specific briefing']).strip() if not pd.isna(row['Specific briefing']) else ""
        expected_output = str(row['Expected output']).strip() if not pd.isna(row['Expected output']) else ""
        
        # Extract rules from general and specific briefings
        rules = []
        if general_topic:
            rules.extend([rule.strip() for rule in general_topic.split('\n') if rule.strip()])
        if specific_briefing:
            rules.extend([rule.strip() for rule in specific_briefing.split('\n') if rule.strip()])
        
        # Create example dictionary
        example = {
            "output": expected_output
        }
        
        # Create template dictionary
        template = {
            "component_type": component_type,
            "element_type": element_type,
            "char_limit": char_limit,
            "examples": [example] if expected_output else [],
            "rules": rules
        }
        
        templates.append(template)
    
    return templates

def main():
    # Get the current file's directory (knowledge_base folder)
    current_dir = Path(__file__).parent
    
    # Input paths
    email_components_excel_path = current_dir.parent / "docs" / "DECATHLON_CRM_Email-copy-elements.xlsx"
    
    # Output path
    templates_output_path = current_dir / "decathlon_template_kb.json"
    
    print(f"Parsing Email Components Excel file: {email_components_excel_path}")
    # Parse Email Components
    templates = parse_email_components_excel(str(email_components_excel_path))
    
    # Add global rules to each template
    global_rules = [
        "Use friendly and inviting tone",
        "Maintain brand voice",
        "Be concise and direct",
        "Focus on customer benefits",
        "Use active voice",
        "Avoid technical jargon"
    ]
    
    for template in templates:
        template['rules'].extend([rule for rule in global_rules if rule not in template['rules']])
    
    # Save templates to JSON
    with open(templates_output_path, 'w', encoding='utf-8') as f:
        json.dump(templates, f, ensure_ascii=False, indent=2)
    
    print(f"Templates saved to: {templates_output_path}")
    print("Done!")

if __name__ == "__main__":
    main()