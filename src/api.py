from __future__ import annotations

from http import HTTPStatus

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings
from .agent import AgentService
from .cache import ResilientCache
from .contracts import (
    ActionResult,
    AuthContext,
    ChatRequest,
    ChatResponse,
    ConfirmActionRequest,
    LeaveRequestView,
    ProblemDetail,
)
from .providers import OpenAIProvider
from .repository import ConflictError, PostgresRepository
from .retrieval import RetrievalService
from .tools import ToolExecutor
from .domain import DomainError
from .security import MedicalPrivacyError


def build_runtime(settings: Settings):
    repository = PostgresRepository(
        database_url=settings.database_url,
        embedding_dimensions=settings.embedding_dimensions,
    )
    cache = ResilientCache(settings.redis_url)

    if not settings.live_model_ready:
        # Keep liveness/readiness available even before model configuration.
        return repository, cache, None

    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        agent_model=settings.agent_model,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
        timeout_seconds=settings.model_timeout_seconds,
    )
    retrieval = RetrievalService(
        repository,
        provider,
        cache,
        settings.retrieval_cache_ttl_seconds,
    )
    tools = ToolExecutor(repository, retrieval, settings)
    agent = AgentService(repository, provider, tools, settings)
    return repository, cache, agent


def create_app(*, settings: Settings | None = None, repository=None, cache=None, agent=None) -> FastAPI:
    """Build the documented API without changing endpoint or response contracts."""
    settings = settings or get_settings()
    if repository is None or cache is None or agent is None:
        runtime = build_runtime(settings)
        if repository is None:
            repository = runtime[0]
        if cache is None:
            cache = runtime[1]
        if agent is None:
            agent = runtime[2]
    app = FastAPI()

    def problem(code: int, detail: str | None = None, headers=None) -> JSONResponse:
        messages = {
            400: "Permintaan tidak valid.",
            401: "Autentikasi diperlukan atau token tidak valid.",
            403: "Anda tidak memiliki izin untuk permintaan tersebut.",
            404: "Sumber daya tidak ditemukan atau tidak dapat diakses.",
            405: "Metode permintaan tidak didukung.",
            409: "Permintaan bertentangan dengan status saat ini. Periksa proposal dan kunci idempotensi.",
            422: "Data permintaan tidak valid. Periksa informasi yang diwajibkan.",
            503: "Layanan belum siap. Silakan coba lagi nanti.",
        }
        title = HTTPStatus(code).phrase if code in HTTPStatus._value2member_map_ else "Request Error"
        body = ProblemDetail(title=title, status=code,
                             detail=detail or messages.get(code, "Permintaan tidak dapat diproses saat ini."))
        return JSONResponse(body.model_dump(exclude_none=True), status_code=code,
                            media_type="application/problem+json", headers=headers)

    @app.exception_handler(StarletteHTTPException)
    async def http_problem(request: Request, exc: StarletteHTTPException):
        # Exception details may contain untrusted input; use fixed messages.
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return problem(exc.status_code, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_problem(request: Request, exc: RequestValidationError):
        # Do not serialize exc.errors(): it includes the rejected input.
        return problem(422)

    @app.exception_handler(DomainError)
    async def domain_problem(request: Request, exc: DomainError):
        return problem(422, "Permintaan tidak memenuhi aturan bisnis atau kebijakan yang berlaku.")

    @app.exception_handler(ConflictError)
    async def conflict_problem(request: Request, exc: ConflictError):
        return problem(409)

    @app.exception_handler(MedicalPrivacyError)
    async def privacy_problem(request: Request, exc: MedicalPrivacyError):
        return problem(403, "Permintaan tidak dapat diproses karena tidak sesuai dengan kebijakan perusahaan.")

    @app.exception_handler(Exception)
    async def internal_problem(request: Request, exc: Exception):
        return problem(500)

    def authenticate(authorization: str | None = Header(default=None)) -> AuthContext:
        if not authorization:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="autentikasi diperlukan",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = authorization.removeprefix("Bearer ").strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="token tidak valid",
                headers={"WWW-Authenticate": "Bearer"},
            )
        authenticate_method = getattr(repository, "authenticate", None)
        auth = authenticate_method(token) if callable(authenticate_method) else None
        if auth is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="token tidak valid",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return auth

    def require_admin(auth: AuthContext = Depends(authenticate)) -> AuthContext:
        if auth.role != "monitoring_admin":
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="akses ditolak")
        return auth

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz(response: Response):
        checks = {"database": False, "schema": False, "index": False}
        try:
            readiness = repository.readiness(settings.policy_index_version)
            checks = {key: readiness.get(key) is True for key in checks}
        except Exception:
            pass
        checks["model"] = bool(agent is not None and (
            settings.live_model_ready or settings.model_mode == "fake"
        ))
        try:
            cache_ready = cache.health() is True
        except Exception:
            cache_ready = False
        headers = {"X-Cache-Status": "ready" if cache_ready else "degraded"}
        if not all(checks.values()):
            missing = ", ".join(key for key, ready in checks.items() if not ready)
            return problem(503, "Layanan belum siap: " + missing + ".", headers=headers)
        response.headers.update(headers)
        return {"status": "ready"}

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        responses={401: {"model": ProblemDetail}},
    )
    def chat(request: ChatRequest, auth: AuthContext = Depends(authenticate)) -> ChatResponse:
        if agent is None:
            raise HTTPException(status_code=503, detail="model belum siap")
        return agent.run(request, auth)

    @app.post(
        "/api/v1/action-proposals/{proposal_id}/confirm",
        response_model=ActionResult,
    )
    def confirm_action(
        proposal_id: str,
        request: ConfirmActionRequest,
        auth: AuthContext = Depends(authenticate),
    ) -> ActionResult:
        proposal_lookup = getattr(repository, "get_proposal", None)
        proposal = proposal_lookup(proposal_id) if callable(proposal_lookup) else None
        if proposal is None or proposal[0] != auth.employee_id:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="proposal tidak ditemukan")
        try:
            return repository.confirm_proposal(
                auth.employee_id, proposal_id, request.idempotency_key
            )
        except ConflictError:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="konflik proposal") from None

    @app.get("/api/v1/leave-requests/me", response_model=list[LeaveRequestView])
    def my_leave_requests(auth: AuthContext = Depends(authenticate)) -> list[LeaveRequestView]:
        return repository.list_leave_requests(auth.employee_id)

    @app.get("/api/v1/traces/{trace_id}")
    def trace(trace_id: str, auth: AuthContext = Depends(authenticate)) -> dict:
        trace_lookup = repository.get_trace(trace_id)
        if trace_lookup is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="trace tidak ditemukan")
        if auth.role != "monitoring_admin" and trace_lookup.get("employee_id") != auth.employee_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="akses ditolak")
        return {key: value for key, value in trace_lookup.items() if key != "message"}

    @app.get("/api/v1/monitoring/summary")
    def monitoring_summary(auth: AuthContext = Depends(require_admin)) -> dict:
        return repository.monitoring_summary()

    @app.get("/api/v1/monitoring/alerts")
    def monitoring_alerts(auth: AuthContext = Depends(require_admin)) -> list | dict:
        summary = repository.monitoring_summary()
        return summary.get("alerts", []) if isinstance(summary, dict) else []

    return app


class _LazyApp:
    def __init__(self) -> None:
        self._app: FastAPI | None = None

    async def __call__(self, scope, receive, send) -> None:
        if self._app is None:
            self._app = create_app()
        await self._app(scope, receive, send)


app = _LazyApp()
