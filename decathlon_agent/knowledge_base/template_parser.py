"""
Tool to parse Excel template into JSON knowledge base.
Run this script when the template file changes.
"""

import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Any

def parse_excel_to_kb(excel_path: str) -> Dict[str, Any]:
    """Parse the Excel template file into a knowledge base dictionary."""
    df = pd.read_excel(excel_path)
    
    # Initialize templates dictionary
    templates = {}
    
    # Iterate through rows and build templates
    for _, row in df.iterrows():
        # Skip rows with missing essential data
        if pd.isna(row['Component']) or pd.isna(row['Text element']):
            continue
        
        # Create component key
        component_key = f"{row['Component']}_{row['Text element']}".lower()
        
        # Extract rules from the row
        rules = []
        if not pd.isna(row['General topic']):
            rules.extend([rule.strip() for rule in str(row['General topic']).split('\n') if rule.strip()])
        
        # Create example dictionary
        example = {
            "briefing": str(row['Specific briefing']) if not pd.isna(row['Specific briefing']) else "",
            "output": str(row['Expected output']) if not pd.isna(row['Expected output']) else "",
            "context": str(row['Component'])  # Using component as context
        }
        
        # Update or create template
        if component_key in templates:
            templates[component_key]['examples'].append(example)
            templates[component_key]['rules'].extend([r for r in rules if r not in templates[component_key]['rules']])
        else:
            templates[component_key] = {
                "component_type": row['Component'],
                "element_type": row['Text element'],
                "char_limit": int(row['Character limitation']) if not pd.isna(row['Character limitation']) else 0,
                "examples": [example],
                "rules": rules
            }
    
    return templates

def main():
    # Get the current file's directory (knowledge_base folder)
    current_dir = Path(__file__).parent
    
    # Input and output paths
    excel_path = current_dir.parent / "DECATHLON_CRM_Email-copy-elements.xlsx"
    output_path = current_dir / "decathlon_template_kb.json"
    
    print(f"Parsing template file: {excel_path}")
    
    # Parse Excel to knowledge base
    kb_dict = parse_excel_to_kb(str(excel_path))
    
    # Add global rules to each template
    global_rules = [
        "Use friendly and inviting tone",
        "Maintain brand voice",
        "Be concise and direct",
        "Focus on customer benefits",
        "Use active voice",
        "Avoid technical jargon"
    ]
    
    for template in kb_dict.values():
        template['rules'].extend([rule for rule in global_rules if rule not in template['rules']])
    
    # Save to JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kb_dict, f, ensure_ascii=False, indent=2)
    
    print(f"Knowledge base saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()