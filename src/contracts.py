from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Outcome(str, Enum):
    answered = "answered"
    action_proposed = "action_proposed"
    escalated = "escalated"
    abstained = "abstained"


class AuthContext(BaseModel):
    employee_id: str
    display_name: str
    role: Literal["employee", "monitoring_admin"] = "employee"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=100)

    @field_validator("message")
    @classmethod
    def strip_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class Citation(BaseModel):
    document_id: str
    title: str
    section: str
    excerpt: str


class ToolCallSummary(BaseModel):
    name: str
    status: Literal["ok", "blocked", "error"]
    duration_ms: int = Field(ge=0)


class ActionProposal(BaseModel):
    proposal_id: str
    action_type: Literal["leave_request"] = "leave_request"
    leave_type: str
    start_date: date
    end_date: date
    business_days: int = Field(gt=0)
    reason: str | None = None
    validation_summary: list[str]
    expires_at: datetime
    requires_confirmation: Literal[True] = True


class UsageSummary(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0)


class ChatResponse(BaseModel):
    answer: str
    outcome: Outcome
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallSummary] = Field(default_factory=list)
    action_proposal: ActionProposal | None = None
    trace_id: str
    usage: UsageSummary


class ConfirmActionRequest(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=100)


class ActionResult(BaseModel):
    request_id: str
    status: Literal["pending_manager_review"] = "pending_manager_review"
    idempotent_replay: bool
    audit_reference: str


class LeaveRequestView(BaseModel):
    request_id: str
    leave_type: str
    start_date: date
    end_date: date
    business_days: int
    status: str
    created_at: datetime


class PolicyChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    title: str
    section: str
    version: str
    effective_date: date
    content: str
    score: float = 0.0


class LeaveBalance(BaseModel):
    leave_type: str
    available_days: int = Field(ge=0)
    as_of_date: date


class PrepareLeaveArgs(BaseModel):
    start_date: date
    end_date: date
    leave_type: Literal["annual", "sick", "parental"]
    reason: str | None = Field(default=None, max_length=300)


class SearchPolicyArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=5)


class TicketArgs(BaseModel):
    category: Literal["policy_question", "leave_exception", "technical_issue", "privacy"]
    summary: str = Field(min_length=5, max_length=500)


class ModelToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, Any]


class ModelTurn(BaseModel):
    response_id: str
    text: str = ""
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    output_items: list[Any] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class ProblemDetail(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    trace_id: str | None = None
