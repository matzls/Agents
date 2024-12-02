# Decathlon CRM Copywriter - Project Plan

## High-Level Summary

This project implements a CRM copywriter system for Decathlon using LangChain/LangGraph. The system consists of four main components:

1. **Core Configuration**
   - Static rules for email components
   - Few-shot examples database
   - Brand tone guidelines

2. **Content Generation**
   - LLM-based generation using few-shot prompts
   - Component-specific content creation
   - Retry mechanism for failed attempts

3. **Validation System**
   - Character limit checks
   - Structure validation against examples
   - Brand voice compliance

4. **Workflow Management**
   - State tracking for components
   - Full email campaign orchestration
   - Output formatting and delivery

The system uses a JSON-based configuration for rules and examples, eliminating the need for complex file parsing. Each component type (introduction, category, CTA, etc.) has its own set of examples and rules, ensuring consistent output across all generated content.

---

# Detailed Implementation Plan

## High-Level Architecture

### 1. Core Configuration
- Email component rules (character limits, types)
- Brand tone requirements
- Few-shot examples store for each component type

### 2. Content Generation System
- Few-shot prompt creation
- Content generation with LLM
- Retry mechanism for failed attempts

### 3. Validation System
- Character limit validation
- Structure validation against examples
- Tone and brand voice compliance

### 4. Workflow Management
- State tracking
- Component generation orchestration
- Output formatting and delivery

## Implementation Details

### 1. Core Configuration Files

```python
# config.py
EMAIL_COMPONENT_RULES = {
    "introduction_copy": {
        "type": "Introduction Copy",
        "char_limit": 400
    },
    "category_title": {
        "type": "Category Title",
        "char_limit": 40
    }
    # ... other components
}

# examples_store.py
COMPONENT_EXAMPLES = {
    "introduction_copy": [
        {
            "topic": "Das Wetter wird wieder kühler...",
            "briefing": "",
            "output": "ok, das war's für dieses Jahr..."
        }
        # ... other examples
    ]
}

BRAND_TONE_REQUIREMENTS = {
    "friendly": ["personal", "welcoming"],
    "enthusiastic": ["positive", "energetic"],
    "customer_oriented": ["helpful", "supportive"],
    "style": ["simple", "direct", "playful"]
}
```

### 2. Content Generation System

```python
# generator.py
class ContentGenerator:
    def __init__(self, llm):
        self.llm = llm
        self.validator = ContentValidator()
        self.examples = COMPONENT_EXAMPLES

    def _create_few_shot_prompt(self, component_type, briefing):
        examples = self.examples.get(component_type, [])
        prompt = f"""Du bist ein Decathlon CRM Copywriter. Erstelle Content basierend auf dem Briefing.
        
Hier sind drei Beispiele für {component_type}:

"""
        for idx, example in enumerate(examples, 1):
            prompt += f"""
Beispiel {idx}:
Briefing: {example['topic']}
Output: {example['output']}
"""

        prompt += f"""
Jetzt erstelle basierend auf diesen Beispielen einen neuen Text für:
Briefing: {briefing}

Beachte dabei:
- Maximale Zeichenlänge: {EMAIL_COMPONENT_RULES[component_type]['char_limit']}
- Stil und Struktur wie in den Beispielen
- Freundlicher, motivierender Ton
"""
        return prompt

    async def generate_component(self, component_type, briefing, max_attempts=3):
        for attempt in range(max_attempts):
            prompt = self._create_few_shot_prompt(component_type, briefing)
            result = await self._generate_and_validate(prompt, component_type)
            if result["status"] == "success":
                return result
        
        return {
            "status": "failed",
            "message": "Max attempts reached",
            "attempts": max_attempts
        }
```

### 3. Validation System

```python
# validators.py
class ContentValidator:
    def __init__(self, rules=EMAIL_COMPONENT_RULES):
        self.rules = rules
        self.examples = COMPONENT_EXAMPLES

    def validate_content(self, content, component_type):
        if not self._validate_length(content, component_type):
            return False, f"Content length exceeds limit for {component_type}"
        
        if not self._validate_structure(content, component_type):
            return False, f"Content structure doesn't match examples"
            
        return True, "Content valid"

    def _validate_length(self, content, component_type):
        max_length = self.rules[component_type]["char_limit"]
        return len(content) <= max_length

    def _validate_structure(self, content, component_type):
        examples = self.examples.get(component_type, [])
        # Structure validation implementation
        return True
```

### 4. Workflow Management

```python
# workflow.py
class CopywriterWorkflow:
    def __init__(self, llm):
        self.generator = ContentGenerator(llm)
        self.state = {
            "components": {},
            "status": "in_progress",
            "validation_history": []
        }

    async def generate_email_component(self, component_type, briefing):
        result = await self.generator.generate_component(component_type, briefing)
        self._update_state(component_type, result)
        return result

    async def generate_full_email(self, campaign_briefing):
        components = [
            "introduction_copy",
            "category_title",
            "category_cta"
        ]
        
        results = {}
        for component in components:
            results[component] = await self.generate_email_component(
                component, 
                campaign_briefing
            )
        
        return results
```

## Implementation Schedule

### Phase 1: Core Setup (Days 1-2)
- [x] Create configuration files
- [x] Set up examples store
- [x] Implement basic validation rules

### Phase 2: Generation System (Days 3-4)
- [ ] Implement ContentGenerator
- [ ] Add few-shot prompting
- [ ] Create basic validation checks

### Phase 3: Enhanced Validation (Days 5-6)
- [ ] Implement structure validation
- [ ] Add tone checking
- [ ] Create comprehensive validation pipeline

### Phase 4: Workflow Integration (Days 7-8)
- [ ] Implement full workflow
- [ ] Add state management
- [ ] Create output formatting

## Testing Strategy

1. **Unit Tests**
   - Validation functions
   - Prompt generation
   - State management

2. **Integration Tests**
   - Full component generation
   - Multi-component workflows
   - Error handling

3. **Example-based Tests**
   - Verify outputs match example styles
   - Check character limits
   - Validate tone requirements

## Next Steps

1. Begin with implementing the core configuration files
2. Set up the basic ContentGenerator class
3. Implement validation system
4. Add workflow management

Would you like to start with any specific component of this implementation?