Based on the code changes and our current state, I'll update the README.md to reflect the current implementation and next steps:

```markdown
# Decathlon CRM Copywriter POC

A proof-of-concept implementation of a CRM copywriting system for Decathlon using LangGraph and LangChain.

## Features

- Graph-based workflow management with LangGraph
- OpenAI integration via LangChain
- TypedDict-based state management
- Content validation with character limits
- Comprehensive logging and error handling
- Generation history tracking with attempt counting
- Multi-component support for different content types

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
# Edit .env and add your OpenAI API key
```

## Usage

Run the script:
```bash
python decathlon_copywriter.py
```

The script will:
1. Process multiple content components (titles, copy, etc.)
2. Generate marketing copy using OpenAI with sport-specific context
3. Validate content against character limits and requirements
4. Track generation attempts and provide detailed feedback
5. Display results and any errors

## Architecture

- **State Management**: Uses TypedDict for state handling with proper typing
- **Workflow**: Implements a directed graph with conditional paths
- **Content Generation**: Leverages OpenAI's GPT-4 model with retry logic
- **Validation**: Checks character limits and content quality
- **History Tracking**: Maintains generation attempts and feedback with proper attempt counting

## Current Implementation

- Sport-specific content generation
- Character limit validation
- Generation attempt tracking
- Proper state management using TypedDict
- Retry logic for API calls
- Detailed feedback system

## Immediate Next Steps

1. **Component Schema Implementation**
   - Create JSON schema for email components
   - Implement component configuration loader
   - Example structure:
     ```json
     {
       "KIT_3_LEFT": {
         "title": {
           "char_limit": 50,
           "required": true,
           "style_rules": ["no_exclamation", "no_ellipsis"]
         },
         "copy": {
           "char_limit": 200,
           "required": true,
           "style_rules": ["no_direct_address"]
         }
       }
     }
     ```

2. **Enhanced Validation System**
   - Add style rule validation
   - Implement tone checking
   - Add component-specific validation rules
   - Enhance feedback quality for regeneration attempts

3. **Content Generation Improvements**
   - Add template support
   - Implement style consistency checks
   - Add context-aware generation
   - Improve prompt engineering based on feedback

## Development Priorities

1. **Component Configuration**
   - Create configuration loader
   - Implement component validation rules
   - Add template support

2. **Validation Enhancement**
   - Implement style rules validation
   - Add tone checking
   - Enhance feedback system

3. **Testing Framework**
   - Add unit tests
   - Create integration tests
   - Implement automated testing pipeline

## Future Extensions

1. **Knowledge Base Integration**
   - Implement structured knowledge base
   - Add sport-specific terminology
   - Create template management

2. **System Improvements**
   - Add human review integration
   - Implement memory management
   - Add performance metrics
   - Enhance error handling

## Current Limitations

- Basic character limit validation only
- No style rule enforcement
- Limited template support
- No persistent storage

## Contributing

Please refer to CONTRIBUTING.md for guidelines on how to contribute to this project.
```

Would you like me to:
1. Add more technical details to any section?
2. Create separate documentation for the component schema?
3. Add implementation examples for the next steps?
4. Create a development roadmap with timelines?