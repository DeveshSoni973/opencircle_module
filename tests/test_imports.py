import pytest
import opencircle_module
from opencircle_module.models import Agent, Message
from opencircle_module.orchestrator import GroupChat
from opencircle_module.registry import AgentRegistry

def test_version():
    """Test that the version string is present and valid."""
    assert hasattr(opencircle_module, "__version__")
    assert isinstance(opencircle_module.__version__, str)

def test_imports():
    """Test that core components can be imported."""
    assert Agent is not None
    assert Message is not None
    assert GroupChat is not None
    assert AgentRegistry is not None
