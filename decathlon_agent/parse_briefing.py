import pandas as pd
from pathlib import Path
import json
from typing import Dict

def parse_briefing_excel(excel_path: str) -> Dict[str, Dict[str, str]]:
    """Parse the Briefing Excel file into a dictionary."""
    # Skip the first 5 rows as they contain header information
    df = pd.read_excel(excel_path, skiprows=5, header=None)
    
    print("Columns in DataFrame:", df.columns.tolist())
    
    briefings = {}
    audience = None
    
    # Process each row
    for _, row in df.iterrows():
        # Skip rows with missing essential data in columns 1 and 2
        if pd.isna(row[1]) or pd.isna(row[2]):
            continue
            
        # Skip header/template rows
        if row[1].lower() in ['element', 'template', 'date']:
            continue
            
        module_element = str(row[1]).strip().lower()
        briefing_text = str(row[2]).strip()
        
        briefings[module_element] = {
            'copy_briefing': briefing_text,
            'audience': audience if audience else 'General'
        }
    
    return briefings

def main():
    # Define paths - corrected path resolution
    current_dir = Path(__file__).parent.resolve()
    docs_dir = current_dir / "docs"
    knowledge_base_dir = current_dir / "knowledge_base"

    # Input and output paths
    briefing_excel_path = docs_dir / "20241104_SchwimmenJahresende_Briefing_clean.xlsx"
    briefings_output_path = knowledge_base_dir / "briefings.json"

    print(f"Looking for Briefing Excel file at: {briefing_excel_path}")

    if not briefing_excel_path.exists():
        raise FileNotFoundError(f"Excel file not found at {briefing_excel_path}. Please ensure the file exists in the 'docs' directory.")

    print(f"Parsing Briefing Excel file: {briefing_excel_path}")
    # Parse the briefing file
    briefings = parse_briefing_excel(str(briefing_excel_path))

    # Save to JSON
    knowledge_base_dir.mkdir(exist_ok=True)
    with open(briefings_output_path, 'w', encoding='utf-8') as f:
        json.dump(briefings, f, ensure_ascii=False, indent=2)

    print(f"Briefings saved to: {briefings_output_path}")

if __name__ == "__main__":
    main()