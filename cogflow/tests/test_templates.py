"""
Unit Tests for CogFlow Templates

Tests for the chain-of-thought templates in cogflow.prompts.cot_templates
"""

import pytest
from cogflow.prompts.cot_templates import (
    PromptTemplate,
    TemplateRegistry,
    COT_BASIC,
    COT_DETAILED,
    COT_MATH,
    COT_ANALYSIS,
    REACT_SYSTEM,
    REACT_STEP,
    REACT_WITH_EXAMPLES,
    get_template_registry,
    get_template,
    format_template,
)


class TestPromptTemplate:
    """Tests for the PromptTemplate class."""
    
    def test_basic_template_creation(self):
        """Test creating a basic template."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}!",
            input_variables=["name"],
            description="A test template"
        )
        assert template.name == "test"
        assert template.input_variables == ["name"]
    
    def test_template_formatting(self):
        """Test template formatting with variables."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}, you are {age} years old.",
            input_variables=["name", "age"],
        )
        result = template.format(name="Alice", age=30)
        assert result == "Hello Alice, you are 30 years old."
    
    def test_missing_variable_raises_error(self):
        """Test that missing variables raise ValueError."""
        template = PromptTemplate(
            name="test",
            template="Hello {name} and {friend}!",
            input_variables=["name", "friend"],
        )
        with pytest.raises(ValueError, match="Missing required variables"):
            template.format(name="Alice")
    
    def test_partial_template(self):
        """Test creating a partial template."""
        template = PromptTemplate(
            name="test",
            template="Hello {name}, you are {age} years old.",
            input_variables=["name", "age"],
        )
        partial = template.partial(name="Bob")
        assert "name" not in partial.input_variables
        assert "age" in partial.input_variables


class TestChainOfThoughtTemplates:
    """Tests for CoT templates."""
    
    def test_cot_basic_template(self):
        """Test basic CoT template formatting."""
        result = COT_BASIC.format(question="What is 2 + 2?")
        assert "What is 2 + 2?" in result
        assert "step by step" in result.lower()
    
    def test_cot_detailed_template(self):
        """Test detailed CoT template formatting."""
        result = COT_DETAILED.format(
            question="What is the capital of France?",
            context="This is a geography question.",
        )
        assert "What is the capital of France?" in result
        assert "This is a geography question." in result
    
    def test_cot_math_template(self):
        """Test math-focused CoT template."""
        result = COT_MATH.format(question="Solve for x: 2x + 5 = 15")
        assert "Solve for x: 2x + 5 = 15" in result
        assert "mathematics" in result.lower() or "math" in result.lower()
    
    def test_cot_analysis_template(self):
        """Test analytical CoT template."""
        result = COT_ANALYSIS.format(
            topic="Climate Change",
            question="What are the main causes?"
        )
        assert "Climate Change" in result
        assert "What are the main causes?" in result


class TestReActTemplates:
    """Tests for ReAct templates."""
    
    def test_react_system_template(self):
        """Test ReAct system template has required components."""
        result = REACT_SYSTEM.format(tools="calculator, search, wikipedia")
        assert "Thought" in result
        assert "Action" in result
        assert "calculator, search, wikipedia" in result
    
    def test_react_step_template(self):
        """Test ReAct step template."""
        result = REACT_STEP.format(
            task="Find the weather in Paris",
            history="Thought: I need to search for weather."
        )
        assert "Find the weather in Paris" in result
        assert "I need to search for weather" in result
    
    def test_react_with_examples_template(self):
        """Test ReAct template with examples."""
        result = REACT_WITH_EXAMPLES.format(
            tools="search, calculator",
            task="What is the population of Tokyo?"
        )
        assert "search, calculator" in result
        assert "What is the population of Tokyo?" in result
        assert "Example" in result


class TestTemplateRegistry:
    """Tests for the TemplateRegistry class."""
    
    def test_registry_creation(self):
        """Test creating a template registry."""
        registry = TemplateRegistry()
        assert registry is not None
    
    def test_default_templates_registered(self):
        """Test that default templates are registered."""
        registry = TemplateRegistry()
        template_names = registry.list_templates()
        assert "cot_basic" in template_names
        assert "cot_detailed" in template_names
        assert "react_system" in template_names
    
    def test_get_template(self):
        """Test getting a template by name."""
        registry = TemplateRegistry()
        template = registry.get("cot_basic")
        assert template is not None
        assert template.name == "cot_basic"
    
    def test_get_nonexistent_template(self):
        """Test getting a non-existent template returns None."""
        registry = TemplateRegistry()
        template = registry.get("nonexistent")
        assert template is None
    
    def test_register_custom_template(self):
        """Test registering a custom template."""
        registry = TemplateRegistry()
        custom = PromptTemplate(
            name="custom_test",
            template="Custom: {input}",
            input_variables=["input"],
        )
        registry.register(custom)
        retrieved = registry.get("custom_test")
        assert retrieved is not None
        assert retrieved.name == "custom_test"
    
    def test_get_by_category(self):
        """Test getting templates by category prefix."""
        registry = TemplateRegistry()
        cot_templates = registry.get_by_category("cot")
        assert len(cot_templates) > 0
        for template in cot_templates:
            assert template.name.startswith("cot")


class TestGlobalFunctions:
    """Tests for global template functions."""
    
    def test_get_template_registry(self):
        """Test getting the global registry."""
        registry = get_template_registry()
        assert isinstance(registry, TemplateRegistry)
    
    def test_get_template_function(self):
        """Test the get_template convenience function."""
        template = get_template("cot_basic")
        assert template is not None
        assert template.name == "cot_basic"
    
    def test_format_template_function(self):
        """Test the format_template convenience function."""
        result = format_template("cot_basic", question="Test question")
        assert "Test question" in result
    
    def test_format_nonexistent_template_raises_error(self):
        """Test that formatting non-existent template raises error."""
        with pytest.raises(ValueError, match="Template not found"):
            format_template("nonexistent_template", arg="value")
