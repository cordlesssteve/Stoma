"""Template management system for reports."""

from typing import Dict, List, Optional
from .base_generator import ReportTemplate
from .research_reports import create_research_landscape_template, create_technology_impact_template


class TemplateManager:
    """Manages report templates."""
    
    def __init__(self):
        """Initialize template manager."""
        self.templates: Dict[str, ReportTemplate] = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default report templates."""
        # Research templates
        research_template = create_research_landscape_template()
        self.templates[research_template.template_id] = research_template
        
        # Technology templates
        tech_template = create_technology_impact_template()
        self.templates[tech_template.template_id] = tech_template
    
    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """Get a template by ID."""
        return self.templates.get(template_id)
    
    def get_all_templates(self) -> List[ReportTemplate]:
        """Get all available templates."""
        return list(self.templates.values())
    
    def add_template(self, template: ReportTemplate):
        """Add a new template."""
        self.templates[template.template_id] = template
    
    def list_templates_by_category(self, category: str) -> List[ReportTemplate]:
        """Get templates by category."""
        return [t for t in self.templates.values() if t.category == category]