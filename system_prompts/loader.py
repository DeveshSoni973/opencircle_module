"""Dynamic system prompt loader."""

from pathlib import Path
from string import Template


class SystemPromptLoader:
    """Load system prompt templates from files."""
    
    def __init__(self, prompts_dir=None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent
        self.dir = Path(prompts_dir)
    
    def load(self, name, **kwargs):
        """Load a template and substitute variables."""
        path = self.dir / f"{name}.txt"
        if not path.exists():
            path = self.dir / "default.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt '{name}' not found in {self.dir}")
        
        template = path.read_text()
        return Template(template).substitute(**kwargs)
    
    def list_available(self):
        """Return available prompt names."""
        return [f.stem for f in self.dir.glob("*.txt")]
    
    def __repr__(self):
        available = ", ".join(self.list_available())
        return f"SystemPromptLoader(dir={self.dir}, prompts=[{available}])"