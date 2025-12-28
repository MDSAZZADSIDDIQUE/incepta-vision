"""
Pydantic models for request/response validation and type safety.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User query")
    page: str = Field(..., min_length=1, description="Page context (e.g., 'sales_comparison')")
    
    @validator('message')
    def validate_message(cls, v):
        """Ensure message is not just whitespace."""
        if not v.strip():
            raise ValueError("Message cannot be empty or whitespace")
        return v.strip()


class VisualSpec(BaseModel):
    """Specification for a single visualization."""
    id: str
    type: str = Field(..., regex="^(kpi|bar|line|table)$")
    label: Optional[str] = None
    column: Optional[str] = None
    agg: Optional[str] = Field(None, regex="^(sum|avg|min|max|count|count_distinct)$")
    format: Optional[str] = Field(None, regex="^(number|currency|percent)$")
    title: Optional[str] = None
    xKey: Optional[str] = None
    yKeys: Optional[List[str]] = None


class ReportSpec(BaseModel):
    """AI-generated report specification."""
    title: str
    narrative_md: str
    visuals: List[VisualSpec]
    followups: List[str] = Field(default_factory=list, max_items=5)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    sql: str
    raw_data: List[List[Any]]
    columns: List[str]
    suggestions: List[str] = Field(default_factory=list)
    report: Optional[ReportSpec] = None
    page: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class ExportPDFRequest(BaseModel):
    """Request model for PDF export."""
    title: str = Field(default="MIS AI Report")
    query: str = ""
    summary: str = ""
    sql: str = ""
    columns: List[str]
    raw_data: List[List[Any]]
    dashboard_png_base64: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool
    time: str
    version: str = "1.0.0"
    services: Dict[str, str] = Field(default_factory=dict)
