I'll create a detailed markdown document outlining the knowledge base implementation plan:

```markdown
# Knowledge Base Implementation Plan for Decathlon CRM Copywriter

## Overview
This document outlines the planned implementation of a comprehensive knowledge base system for the Decathlon CRM Copywriter, including structure, loading mechanisms, and integration points.

## 1. Knowledge Base Structure

### Base Structure
```python
knowledge_base = {
    'component_specs': {
        'subject_line': {
            'char_limit': 50,
            'required': True,
            'allow_emojis': True,
            'allow_special_chars': False
        },
        'preheader': {...},
        'headline': {...},
        'body_copy': {...}
    },
    'brand_voice': {
        'tone': ['freundlich', 'einladend', 'begeistert', ...],
        'style': ['einfach und direkt', 'verständliche Sprache', ...],
        'prohibited': ['aggressive Verkaufssprache', ...]
    },
    'structural_rules': {...},
    'sport_contexts': {...},
    'templates': {...},
    'seasonal_contexts': {...}
}
```

## 2. Implementation Components

### 2.1 Knowledge Base Loader
```python
def load_knowledge_base(guidelines_path: str) -> Dict[str, Any]:
    """
    Load and structure brand guidelines and email component specifications
    """
    knowledge_base = {
        # Base structure as shown above
    }

    try:
        with open(guidelines_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            # Parse and update knowledge_base
            logger.info("Successfully loaded guidelines")
    except FileNotFoundError:
        logger.warning(f"Guidelines file not found at {guidelines_path}")
    except Exception as e:
        logger.error(f"Error loading guidelines: {str(e)}")

    return knowledge_base
```

### 2.2 State Integration
```python
class CopyState(TypedDict):
    component: Dict[str, Any]
    generated_content: str
    validation_results: Dict[str, Any]
    errors: List[str]
    attempt_count: int
    status: str
    generation_history: List[Dict[str, Any]]
    knowledge_base: Dict[str, Any]  # New field
```

### 2.3 Enhanced Content Generation
```python
def generate_content(state: CopyState) -> Dict[str, Any]:
    kb = state.get("knowledge_base", {})
    sport = state['component']['audience']
    sport_context = kb.get('sport_contexts', {}).get(sport, {})
    
    prompt = f"""
    [Previous prompt content...]
    
    SPORT-SPEZIFISCHER KONTEXT:
    Equipment: {', '.join(sport_context.get('equipment', []))}
    Aktivitäten: {', '.join(sport_context.get('activities', []))}
    Benefits: {', '.join(sport_context.get('benefits', []))}
    """
    # Rest of the function
```

## 3. Example Content

### 3.1 Sport Contexts
```json
{
    "Schwimmen": {
        "equipment": ["Schwimmbrille", "Badeanzug", "Badehose", "Schwimmflossen"],
        "activities": ["Schwimmtraining", "Wassergymnastik", "Aquafitness"],
        "benefits": ["Ganzkörpertraining", "Ausdauer", "gelenkschonend"]
    },
    "Basketball": {
        "equipment": ["Basketball", "Basketballschuhe", "Trikot"],
        "activities": ["Streetball", "Teamtraining", "Korbwürfe"],
        "benefits": ["Teamspirit", "Koordination", "Schnelligkeit"]
    }
}
```

### 3.2 Templates
```json
{
    "subject_line": [
        "🏃‍♂️ {sport}: Neue {product_category} entdecken",
        "{benefit} mit unserer {product_category}"
    ],
    "headline": [
        "{sport}-Saison: {key_benefit}",
        "Entdecke {product_category} für {sport}"
    ]
}
```

## 4. Implementation Steps

1. **File Structure Setup**
   - Create knowledge base directory
   - Set up separate files for different content types
   - Implement file loading mechanism

2. **Content Organization**
   - Organize existing guidelines into structured format
   - Create sport-specific content sections
   - Define templates and patterns

3. **Integration**
   - Update state management
   - Enhance content generation
   - Add validation rules

4. **Validation System**
   - Implement content validation against knowledge base rules
   - Add sport-specific validation
   - Check template compliance

## 5. Future Enhancements

1. **Dynamic Updates**
   - API for updating knowledge base
   - Version control for content

2. **Content Analytics**
   - Track template usage
   - Measure content effectiveness

3. **Advanced Features**
   - AI-powered template generation
   - Automatic content optimization

## 6. Code Integration Example

```python
def main():
    # Load knowledge base
    kb_path = "path/to/knowledge_base/"
    knowledge_base = load_knowledge_base(kb_path)
    
    # Initialize components with knowledge base
    test_components = [
        CopyComponent(
            name="headline basic",
            char_limit=400,
            briefing="...",
            audience="Schwimmen"
        )
    ]
    
    for component in test_components:
        initial_state = initialize_state(
            component.to_dict(), 
            knowledge_base=knowledge_base
        )
        workflow = create_workflow()
        final_state = workflow.invoke(initial_state)
```

## 7. Testing Strategy

1. **Unit Tests**
   - Knowledge base loading
   - Content validation
   - Template processing

2. **Integration Tests**
   - Full workflow with knowledge base
   - Error handling
   - State management

3. **Content Tests**
   - Template validation
   - Sport-specific content generation
   - Character limit compliance

## Notes

- Keep knowledge base files in version control
- Regular updates to sport-specific content
- Monitor and log knowledge base usage
- Consider implementing caching for frequently used content

```

This markdown document provides a comprehensive plan for implementing the knowledge base system. Would you like me to:
1. Add more specific examples for any section?
2. Expand on the implementation steps?
3. Add more detail to the testing strategy?
4. Create separate documents for specific components?