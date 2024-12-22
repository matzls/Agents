# Decathlon Copy Generation Agent

This project implements an agent for generating marketing copy for Decathlon CRM emails. It uses Langchain and LangGraph to orchestrate a workflow involving content generation, validation, and formatting.

## Overview

The agent takes a list of `CopyComponent` objects as input, each defining the requirements for a specific piece of content (e.g., headline, body text). It then uses an OpenAI language model to generate content for each component, validates the generated content against predefined rules (like character limits), and formats the final output into a JSON file.

The workflow follows a specific pattern:
1. **Generate:** Creates content for each component
2. **Validate:** Checks if the content meets all requirements
3. **Format Output:** Prepares the final output with either successful content or error messages

## Features

- **Component-based Content Generation:** Generates content based on predefined components with specific requirements
- **LLM Integration:** Supports multiple language models:
  - OpenAI's GPT models
  - Google's Gemini models (via ChatGoogleGenerativeAI)
- **Validation System:** Ensures generated content adheres to:
  - Character limits
  - Content formatting rules
  - Empty content checks
- **Retry Logic:** 
  - Configurable max attempts per component
  - Automatic retries with feedback from previous attempts
- **Structured Output:** 
  - JSON format with clear status indicators
  - Detailed error messages when generation fails
  - Timestamp-based file naming
- **Logging:** Comprehensive logging system for debugging and monitoring

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file with the required API keys:
   ```
   OPENAI_API_KEY="your_openai_api_key"
   GOOGLE_API_KEY="your_google_api_key"  # If using Gemini
   ```

5. **Prepare the knowledge base:**
   Ensure `template_kb_full_notokens.json` is present in the project directory with your content examples and guidelines.

## Usage

### Basic Example

```python
from decathlon_studio import CopyComponent, graph

# Define your components
components = [
    CopyComponent(
        component_type="headline_basic",
        element_type="headline_text",
        char_limit=50,
        audience="sports_enthusiasts",
        briefing="Focus on running equipment"
    )
]

# Create initial state
initial_state = {
    "components": components,
    "generated_content": [],
    "validation_results": [],
    "errors": [],
    "attempt_count": 0,
    "status": "starting",
    "generation_history": [],
    "output": []
}

# Run the workflow
result = await graph.ainvoke(initial_state)
```

### Output Structure

The generated output will be saved in the `exports` directory with the following structure:
```json
{
    "timestamp": "2024-03-XX...",
    "status": "success|failed",
    "components": [
        {
            "component_type": "headline_basic",
            "element_type": "headline_text",
            "audience": "sports_enthusiasts",
            "status": "success|failed",
            "content": "Generated content or error message",
            "errors": []
        }
    ],
    "generation_history": []
}
```

## Workflow Graph

The agent uses a LangGraph workflow with the following nodes:
- `START` → `generate`
- `generate` → `validate`
- `validate` → `generate` (if validation fails and max attempts not reached)
- `validate` → `format_output` (if validation passes or max attempts reached)
- `format_output` → `END`

## Error Handling

The system handles various error cases:
- Generation failures
- Validation failures
- API errors
- Character limit violations

All errors are documented in the output file, making debugging and monitoring straightforward.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.