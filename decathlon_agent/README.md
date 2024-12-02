# Decathlon CRM Copywriter POC

A proof-of-concept implementation of a CRM copywriting system for Decathlon using LangGraph and LangChain.

## Features

- Graph-based workflow management with LangGraph
- OpenAI integration via LangChain
- Content validation with character limits
- Error handling and logging
- Type-safe implementation with TypedDict
- Extensible architecture for future enhancements

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
1. Initialize a test component for the Autumn Sports Collection
2. Generate marketing copy using OpenAI
3. Validate the content against requirements
4. Display results and any errors

## Architecture

- **State Management**: Uses TypedDict for type-safe state handling
- **Workflow**: Implements a directed graph with conditional paths
- **Content Generation**: Leverages OpenAI's GPT models via LangChain
- **Validation**: Checks character limits and content quality

## Future Extensions

- Human review integration
- Memory management for context awareness
- Multiple component handling
- Advanced validation rules
- Performance metrics tracking 