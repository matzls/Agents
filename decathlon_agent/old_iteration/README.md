Looking at your screen share, I can see your Git commit history which shows several important updates including character and token limit handling, gitignore additions, and README changes. Based on these recent changes, I'll provide an updated README that incorporates all the latest features and changes.

Here's the complete updated README.md:

```markdown
# Decathlon CRM Copywriter

A production-ready implementation of a CRM copywriting system for Decathlon using LangGraph and LangChain.

## Project Structure

```plaintext
decathlon_agent/
├── knowledge_base/
│   ├── template_kb.py         # Knowledge base classes and loading functions
│   ├── template_parser.py     # Excel to JSON parser for templates
│   ├── decathlon_template_kb.json  # Generated template knowledge base
│   └── briefings.json        # Generated briefings knowledge base
├── docs/
│   ├── DECATHLON_CRM_Email-copy-elements.xlsx  # Email component templates
│   └── 20241104_SchwimmenJahresende_Briefing_clean.xlsx  # Briefing data
├── studio/                    # LangGraph Studio compatible versions
│   └── decathlon_copywriter_agent_dev_studio.py
├── exports/                   # Generated copy outputs
│   └── copy_export_[timestamp]_[component].json
├── parse_briefing.py         # Briefing Excel to JSON parser
├── generate_test_components.py # Test component generator
├── test_components.json      # Test component configurations
├── decathlon_copywriter.py    # Stable production version
├── decathlon_copywriter_dev.py # Development version with latest features
├── .env                      # Environment configuration
├── .gitignore               # Git configuration
└── requirements.txt          # Project dependencies
```

## Features

### Core Features
- Graph-based workflow management with LangGraph
- OpenAI GPT-4 integration via LangChain
- Pydantic-based state management
- Structured knowledge base for templates and examples
- Component-based content generation with strict character and token limits
- Enhanced validation system with centralized functions
- Improved duplicate content detection
- Export functionality for generated content

### Knowledge Base
- Excel-based template and briefing management
- Structured component templates and briefings
- Example-based generation
- Sport-specific rules and guidelines
- Automated JSON conversion for both templates and briefings
- Support for multiple briefing types and audiences

### Validation System
- Dual validation: character and token limits
- Centralized cleaning and validation functions
- Content quality checks including length and emptiness
- Generation attempt tracking with comprehensive error handling
- Enhanced feedback system with specific character and token count guidance
- Cached token counting for performance
- Duplicate content detection across generation attempts
- Smart regeneration logic based on validation results

### Export System
- JSON-based export format with timestamps
- Component-specific export files
- Complete generation history
- Detailed validation results including token metrics
- Structured metadata

### Testing and Development
- Test component generation utilities
- Configurable test cases via JSON
- Development environment with latest features
- Studio integration for visual workflow development

## Setup

1. Create and activate a virtual environment:
```bash
python3 -m venv decathlon_venv
source decathlon_venv/bin/activate  # On Unix/macOS
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
# Parse email component templates
python knowledge_base/template_parser.py

# Parse briefing data
python parse_briefing.py
```

5. Generate test components (optional):
```bash
python generate_test_components.py
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

### LangGraph Studio Version
The studio version accepts input in the following format:
```json
{
  "components": [
    {
      "name": "swimming_title",
      "char_limit": 30,
      "token_limit": 10,
      "briefing": "Ab ins Wasser - kurze, knackige Motivation",
      "audience": "Schwimmen",
      "component_type": "headline",
      "element_type": "title"
    }
  ],
  "max_attempts": 3
}
```

Example usage in LangGraph Studio:
1. Load the agent file (decathlon_copywriter_agent_studio.py)
2. Paste the input JSON into the Input section
3. Click Submit to start the generation process

The system will:
1. Generate sport-specific marketing copy
2. Validate against requirements (character and token limits, content rules)
3. Export results to JSON files
4. Provide detailed validation feedback

## Technical Details

### State Management
```python
@dataclass(frozen=True)
class CopyComponent:
    component_type: str
    element_type: str
    char_limit: int
    token_limit: int
    audience: str
    max_attempts: int = 3

class AgentState(TypedDict):
    component: CopyComponent
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]
```

### Validation Functions
```python
def validate_limits(content: str, char_limit: int, token_limit: int) -> Dict[str, Any]
def clean_content(content: str) -> str
def generate_feedback(validation_results: Dict[str, Any], char_limit: int, token_limit: int) -> List[str]
```

### Export Format
```json
{
  "timestamp": "ISO-8601 timestamp",
  "status": "status_code",
  "total_attempts": 0,
  "components": [
    {
      "name": "component_name",
      "content": "generated_text",
      "validation": {
        "char_count": 000,
        "token_count": 000,
        "within_char_limit": true/false,
        "within_token_limit": true/false,
        "is_empty": true/false
      }
    }
  ],
  "generation_history": []
}
```

## Development Status

### Recently Completed
- Token limit implementation alongside character limits
- Centralized validation functions
- Enhanced export file naming and organization
- Pydantic model integration
- Improved duplicate content detection
- Smart regeneration logic
- Enhanced feedback system
- Proper state management with TypedDict

### In Progress
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

4. LangGraph Studio Integration
   - Update studio version with latest features
   - Implement new validation system
   - Add duplicate detection
   - Enhance state management

### TODOs
1. Update LangGraph Studio file with:
   - New validation system
   - Duplicate content detection
   - Enhanced state management
   - Improved feedback system
   - Smart regeneration logic

2. Performance Optimization
   - Token counting optimization
   - State management efficiency
   - Reduced API calls

3. Documentation
   - Update API documentation
   - Add more code examples
   - Improve setup instructions

## Current Limitations

- Single language support (German)
- Sequential processing only
- Limited style rule enforcement
- Basic error recovery
- LangGraph Studio version requires update to match current features
- Knowledge base integration limited in Studio version

## Contributing

1. Use the development version (`decathlon_copywriter_dev.py`) for new features
2. Maintain knowledge base structure when adding components
3. Update tests for new functionality
4. Document changes in code and README

## License

Proprietary - All rights reserved
```

This updated README reflects:
1. The new Pydantic-based state management
2. Added token management features
3. Updated export format with token metrics
4. Streamlined validation functions
5. More detailed technical specifications
6. Current export file naming convention
7. Recent Git changes visible in your commit history



TODO:
- check if the character limit generated in the component is really matching the character limit of the email components. I go the feeling the matching of the elements form email_components and the briefing file is not really working yet. 