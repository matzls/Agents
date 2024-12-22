# Documentation for Decathlon Copy Generation Agent

This documentation provides a detailed overview of the Decathlon Copy Generation Agent, including its architecture, components, and usage.

## Overview

The Decathlon Copy Generation Agent is designed to automate the creation of marketing copy for Decathlon CRM emails. It utilizes a state-based graph workflow built with Langchain and LangGraph, leveraging OpenAI's language models. The agent processes predefined content components, generates corresponding text, validates against specific criteria, and outputs structured JSON.

## Architecture

The agent's architecture is based on a LangGraph `StateGraph` with the following key nodes:

1. **`generate_content`:** Generates content for each `CopyComponent` using:
   - OpenAI's GPT-4 model (gpt-4o-mini)
   - Retry logic for robustness
   - Dynamic prompt construction based on component specifications
   - Feedback incorporation from previous attempts

2. **`validate_content`:** Validates generated content against:
   - Character limits
   - Empty content checks
   - Unwanted prefix detection
   - Content quality rules

3. **`format_output`:** Structures the final output into JSON and saves to the exports directory.

## Key Components

### `CopyComponent`

```python
@dataclass
class CopyComponent:
    component_type: str
    element_type: str
    char_limit: int
    audience: str
    max_attempts: int = 3
    briefing: str = None
```

### `State`

```python
class State(TypedDict):
    components: List[CopyComponent]
    generated_content: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]
    output: List[Dict[str, Any]]
```

## Core Functions

### Content Generation

```python
async def generate_content(state: State) -> State:
```
- Manages the content generation process
- Loads knowledge base
- Constructs prompts with feedback
- Handles generation errors
- Updates state with generated content

### Content Validation

```python
def validate_content(state: State) -> State:
```
- Validates character limits
- Checks for empty content
- Detects unwanted prefixes
- Generates validation feedback
- Updates state with validation results

### Output Formatting

```python
def format_output(state: State) -> State:
```
- Structures final output
- Includes component details
- Adds validation status
- Saves to timestamped JSON file

## Workflow Process

1. **Initialization:**
   - Load environment variables
   - Initialize OpenAI model
   - Set up logging

2. **Generation Phase:**
   - Process each component
   - Construct dynamic prompts
   - Generate content with retry logic
   - Clean generated content

3. **Validation Phase:**
   - Check character limits
   - Validate content quality
   - Generate feedback if needed

4. **Decision Making:**
   - Continue to formatting if validation passes
   - Retry generation with feedback if validation fails
   - End workflow if max attempts reached

5. **Output Phase:**
   - Format results into JSON
   - Save to exports directory
   - Include generation history

## Configuration

### Environment Setup
```
OPENAI_API_KEY="your-api-key"
```

### Model Configuration
```python
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.9,
    max_retries=2,
    request_timeout=30
)
```

### Logging
```python
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
```

## Output Structure

The generated JSON output includes:
- Timestamp
- Overall status
- Component details
  - Component type
  - Element type
  - Audience
  - Generated content
  - Validation status
  - Any error messages
- Generation history

## Error Handling

- Retry mechanism for API calls
- Detailed error logging
- Component-specific error tracking
- Maximum attempt limits
- Validation feedback generation

## File Structure

```
exports/
    copy_export_YYYYMMDD_HHMMSS.json
template_kb_full_notokens.json
.env
```

## Best Practices

1. **Prompt Engineering:**
   - Clear instructions
   - Audience-specific guidance
   - Length constraints
   - Brand voice guidelines

2. **Content Validation:**
   - Character limit checks
   - Quality assurance
   - Format verification

3. **Error Management:**
   - Graceful failure handling
   - Detailed error messages
   - Retry mechanisms

## Limitations

- Dependent on OpenAI API availability
- Maximum retry attempts fixed at 3
- Synchronous validation process
- Fixed output format

## Future Enhancements

- Async validation processing
- Additional validation rules
- Custom retry strategies
- Enhanced feedback mechanisms
- Integration with CMS systems
```
