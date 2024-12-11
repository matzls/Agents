I'll update the README.md to reflect the current state of the project, including the development version and knowledge base integration. Here's the updated version:

```markdown
# Decathlon CRM Copywriter

A production-ready implementation of a CRM copywriting system for Decathlon using LangGraph and LangChain.

## Project Structure

```plaintext
decathlon_agent/
├── knowledge_base/
│   ├── template_kb.py         # Knowledge base classes and loading functions
│   ├── template_parser.py     # Excel to JSON parser for templates
│   └── decathlon_template_kb.json  # Generated knowledge base
├── exports/                   # Generated copy outputs
├── decathlon_copywriter.py    # Stable production version
├── decathlon_copywriter_dev.py # Development version with latest features
├── .env                      # Environment configuration
└── requirements.txt          # Project dependencies
```

## Features

### Core Features
- Graph-based workflow management with LangGraph
- OpenAI GPT-4 integration via LangChain
- TypedDict-based state management
- Structured knowledge base for templates and examples
- Component-based content generation
- Export functionality for generated content

### Knowledge Base
- Excel-based template management
- Structured component templates
- Example-based generation
- Sport-specific rules and guidelines
- Automated JSON conversion

### Validation System
- Character limit validation
- Content quality checks
- Generation attempt tracking
- Comprehensive error handling
- Detailed feedback system

### Export System
- JSON-based export format
- Timestamp-based file naming
- Complete generation history
- Validation results included
- Structured metadata

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

4. Initialize knowledge base:
```bash
python knowledge_base/template_parser.py
```

## Usage

### Production Version
```bash
python decathlon_copywriter.py
```

### Development Version
```bash
python decathlon_copywriter_dev.py
```

The system will:
1. Load the knowledge base
2. Process content components
3. Generate sport-specific marketing copy
4. Validate against requirements
5. Export results to JSON files

## Component Types

Currently supported components:
- Headline Basic (Introduction Copy)
- Category Lifestyle CTA
- Advice (Title, Copy, CTA)
- Product (Title, Subtitle)
- Banner (Headline, Copy, CTA)

Each component includes:
- Character limits
- Example texts
- Specific rules
- Validation requirements

## Development Status

### Recently Completed
- Knowledge base integration
- Template-based generation
- Export functionality
- Component type system
- Example-based learning

### In Progress
- Enhanced validation rules
- Style consistency checks
- Performance optimization
- Error handling improvements

### Planned Features
1. Style Rule Enforcement
   - Tone consistency
   - Brand voice validation
   - Language style checks

2. Template Enhancement
   - Dynamic template selection
   - Context-aware generation
   - Improved example matching

3. Export System Enhancement
   - Multiple export formats
   - Batch processing
   - Export aggregation

## Technical Details

### State Management
```python
@dataclass(frozen=True)
class CopyComponent:
    name: str
    char_limit: int
    briefing: str
    audience: str
    component_type: str
    element_type: str
    max_attempts: int = 3
    url: Optional[str] = None
```

### Knowledge Base Structure
```python
class ComponentTemplate:
    component_type: str
    element_type: str
    char_limit: int
    examples: List[ComponentExample]
    rules: List[str]
```

### Export Format
```json
{
  "timestamp": "ISO-8601 timestamp",
  "component": {
    "name": "component_name",
    "type": "component_type",
    "element": "element_type",
    "audience": "target_audience",
    "char_limit": 000,
    "briefing": "briefing_text"
  },
  "generation_result": {
    "final_content": "generated_text",
    "char_count": 000,
    "status": "status_code",
    "total_attempts": 0
  },
  "generation_history": [
    {
      "attempt": 1,
      "content": "generated_text",
      "feedback": "validation_feedback",
      "validation": {
        "char_count": 000,
        "within_limit": true/false,
        "is_empty": true/false
      }
    }
  ]
}
```

## Contributing

1. Use the development version (`decathlon_copywriter_dev.py`) for new features
2. Maintain knowledge base structure when adding components
3. Update tests for new functionality
4. Document changes in code and README

## Current Limitations

- Single language support (German)
- Sequential processing only
- Limited style rule enforcement
- Basic error recovery

## License

Proprietary - All rights reserved
```

Would you like me to:
1. Add more technical documentation for any specific feature?
2. Create separate documentation for the knowledge base system?
3. Add more example configurations?
4. Detail the development workflow?