"""FastAPI REST server mirroring Pinecone's HTTP API surface.

Endpoints implemented (Pinecone-shaped):

    # Data plane (per-index).
    POST /vectors/upsert
    POST /query                          (Pinecone's /query)
    POST /vectors/fetch
    POST /vectors/delete
    POST /vectors/update
    POST /vectors/list
    POST /describe_index_stats
    GET  /health

    # Control plane (server-wide).
    POST   /indexes
    GET    /indexes
    GET    /indexes/{name}
    DELETE /indexes/{name}

Authentication: pass ``--api-key KEY`` (or env ``STM_API_KEY``); every
client request must include ``Api-Key: KEY`` in the headers. The
default is no auth (open) for local use.

Index scoping: in this single-process server, indexes are subdirectories
under ``--path`` and clients address them by ``X-Index-Name`` header.
That's a simpler mapping than Pinecone's per-index hostname scheme but
preserves the same wire surface.

Run::

    python -m spectraltm_db serve --path /var/stm --api-key dev
"""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

log = logging.getLogger(__name__)


# ---- Pydantic request models -----------------------------------------------
# Defined at module level so Pydantic v2 resolves ForwardRef cleanly
# when they're used as FastAPI body parameters inside ``build_app``.

try:
    from pydantic import BaseModel, Field
except ImportError:                                                # pragma: no cover
    BaseModel = None  # type: ignore
    Field = None     # type: ignore


class _Vector(BaseModel):
    id: str
    values: List[float]
    metadata: dict[str, Any] | None = None
    sparse_values: dict[str, List[float]] | None = None   # v0.2; ignored


class UpsertReq(BaseModel):
    vectors: List[_Vector]
    namespace: str | None = None


class QueryReq(BaseModel):
    vector: List[float] | None = None
    sparse_vector: List[float] | None = None
    top_k: int = 10
    filter: dict[str, Any] | None = None
    include_metadata: bool = True
    include_values: bool = False
    namespace: str | None = None


class FetchReq(BaseModel):
    ids: List[str]
    namespace: str | None = None


class DeleteReq(BaseModel):
    ids: List[str] | None = None
    delete_all: bool | None = None
    filter: dict[str, Any] | None = None
    namespace: str | None = None


class UpdateReq(BaseModel):
    id: str
    values: List[float] | None = None
    set_metadata: dict[str, Any] | None = None
    namespace: str | None = None


class ListReq(BaseModel):
    prefix: str | None = None
    limit: int | None = 100
    namespace: str | None = None


class StatsReq(BaseModel):
    namespace: str | None = None


class CreateIndexReq(BaseModel):
    name: str
    dimension: int
    metric: str = "cosine"
    spec: dict[str, Any] | None = None


# ---- Errors -> HTTP status mapping -----------------------------------------

def _http_error(status: int, message: str, code: str = "INVALID_ARGUMENT"):
    try:
        from fastapi import HTTPException
    except ImportError:                                            # pragma: no cover
        return RuntimeError(f"HTTP {status}: {message}")
    return HTTPException(status_code=status,
                         detail={"error": {"code": code, "message": message}})


def _invalid(message: str):
    return _http_error(400, message, "INVALID_ARGUMENT")


def _not_found(message: str):
    return _http_error(404, message, "NOT_FOUND")


def _unauthorized():
    return _http_error(401, "invalid or missing Api-Key", "UNAUTHENTICATED")


# ---- Index registry ---------------------------------------------------------

class IndexRegistry:
    """Filesystem-backed registry of `name -> Index` instances."""

    def __init__(
        self, data_root: str | Path,
        *, default_dimension: int = 384,
        default_compression: str = "spectral_k64",
    ) -> None:
        self._root = Path(data_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._default_dimension = default_dimension
        self._default_compression = default_compression
        self._cache: dict[str, Any] = {}

    @property
    def root(self) -> Path:
        return self._root

    def list_indexes(self) -> list[dict]:
        out = []
        if not self._root.exists():
            return out
        for p in sorted(self._root.iterdir()):
            if not p.is_dir():
                continue
            if not (p / "config.json").exists():
                continue
            try:
                idx = self._get(p.name)
                s = idx.describe_index_stats()
                out.append({
                    "name": p.name,
                    "dimension": s["dimension"],
                    "metric": s["metric"],
                    "compression": s["compression"],
                    "total_vector_count": s["total_vector_count"],
                })
            except Exception:
                continue
        return out

    def get_or_load(self, name: str) -> Any:
        return self._get(name)

    def create(
        self, name: str, *, dimension: int, metric: str = "cosine",
        compression: str | None = None, calibration_sample=None,
    ) -> Any:
        from .index import Index
        path = self._root / name
        if (path / "config.json").exists():
            raise FileExistsError(f"index {name!r} already exists at {path}")
        compression = compression or self._default_compression
        if calibration_sample is None:
            import random
            rng = random.Random(name)
            calibration_sample = [rng.gauss(0, 1) for _ in range(200 * dimension)]
        return Index.create(
            name=name, path=str(path),
            dimension=dimension, metric=metric, compression=compression,
            calibration_sample=calibration_sample,
        )

    def delete(self, name: str) -> None:
        import shutil
        path = self._root / name
        if path.exists():
            self._cache.pop(name, None)
            shutil.rmtree(path, ignore_errors=True)

    def _get(self, name: str) -> Any:
        from .index import Index
        from .errors import NotFound
        idx = self._cache.get(name)
        if idx is not None:
            return idx
        path = self._root / name
        if not (path / "config.json").exists():
            raise NotFound(f"no index named {name!r}")
        idx = Index.open(str(path))
        self._cache[name] = idx
        return idx


# ---- Auth ------------------------------------------------------------------

class APIKeyAuth:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("STM_API_KEY") or ""

    def is_unauth(self) -> bool:
        return self.api_key == ""

    def check(self, header_value: str | None) -> bool:
        if self.is_unauth():
            return True
        if not header_value:
            return False
        return secrets.compare_digest(header_value, self.api_key)


# ---- App builder -----------------------------------------------------------

def build_app(
    data_root: str | Path,
    *,
    api_key: str | None = None,
) -> Any:
    """Build the FastAPI app. Imported lazily."""
    try:
        from fastapi import Body, FastAPI, Header, Request
        from fastapi.responses import JSONResponse
    except ImportError as e:
        raise ImportError(
            "REST server requires fastapi + uvicorn. Install with "
            "`pip install \"spectraltm-db[server]\"`."
        ) from e

    auth = APIKeyAuth(api_key)
    registry = IndexRegistry(data_root)

    @asynccontextmanager
    async def lifespan(_app):
        yield

    app = FastAPI(
        title="spectraltm-db",
        description="Pinecone-compatible REST server backed by Spectral encoding.",
        version="0.1.1",
        lifespan=lifespan,
    )

    def _resolve_index_name(name: str | None) -> str:
        if not name:
            raise _invalid("X-Index-Name header is required")
        return name

    # ---- Data plane ----------------------------------------------------

    @app.post("/vectors/upsert")
    async def vectors_upsert(
        req: UpsertReq = Body(...),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        idx_name = _resolve_index_name(x_index_name)
        try:
            idx = registry.get_or_load(idx_name)
        except _Nf:
            raise _not_found(f"index {idx_name!r} not found")
        r = idx.upsert(
            vectors=[v.model_dump() for v in req.vectors],
            namespace=req.namespace,
        )
        return {"upserted_count": r["upserted_count"]}

    @app.post("/query")
    async def vectors_query(
        req: QueryReq = Body(...),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        if req.vector is None:
            raise _invalid("query requires a `vector`")
        try:
            idx = registry.get_or_load(_resolve_index_name(x_index_name))
        except _Nf:
            raise _not_found(f"index not found")
        try:
            r = idx.query(
                vector=req.vector, top_k=req.top_k,
                filter=req.filter, namespace=req.namespace,
                include_metadata=req.include_metadata,
                include_values=req.include_values,
            )
        except Exception as e:
            raise _invalid(str(e))
        return r

    @app.post("/vectors/fetch")
    async def vectors_fetch(
        req: FetchReq = Body(...),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        try:
            idx = registry.get_or_load(_resolve_index_name(x_index_name))
        except _Nf:
            raise _not_found(f"index not found")
        return idx.fetch(ids=req.ids, namespace=req.namespace)

    @app.post("/vectors/delete")
    async def vectors_delete(
        req: DeleteReq = Body(...),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        try:
            idx = registry.get_or_load(_resolve_index_name(x_index_name))
        except _Nf:
            raise _not_found(f"index not found")
        return idx.delete(
            ids=req.ids, filter=req.filter,
            delete_all=bool(req.delete_all), namespace=req.namespace,
        )

    @app.post("/vectors/update")
    async def vectors_update(
        req: UpdateReq = Body(...),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        try:
            idx = registry.get_or_load(_resolve_index_name(x_index_name))
        except _Nf:
            raise _not_found(f"index not found")
        return idx.update(
            id=req.id, values=req.values,
            set_metadata=req.set_metadata, namespace=req.namespace,
        )

    @app.post("/vectors/list")
    async def vectors_list(
        req: ListReq = Body(...),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        try:
            idx = registry.get_or_load(_resolve_index_name(x_index_name))
        except _Nf:
            raise _not_found(f"index not found")
        ids = []
        for ns in (idx.list_namespaces() or ["default"]):
            if req.namespace and ns != req.namespace:
                continue
            handle = idx._ns_cache.get(ns)    # noqa: SLF001
            if handle is None:
                continue
            ids.extend(handle.id_map.all_ids())
        if req.prefix:
            ids = [i for i in ids if i.startswith(req.prefix)]
        if req.limit is not None:
            ids = ids[: req.limit]
        return {"ids": ids, "namespace": req.namespace}

    @app.post("/describe_index_stats")
    async def describe_index_stats(
        req: StatsReq = Body(default_factory=StatsReq),
        x_index_name: str | None = Header(default=None),
    ) -> dict:
        from .errors import NotFound as _Nf
        try:
            idx = registry.get_or_load(_resolve_index_name(x_index_name))
        except _Nf:
            raise _not_found(f"index not found")
        return idx.describe_index_stats(namespace=req.namespace)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "version": "0.1.1"}

    # ---- Control plane --------------------------------------------------

    @app.post("/indexes")
    async def create_index(req: CreateIndexReq = Body(...)) -> dict:
        try:
            compression = (req.spec or {}).get("compression") or None
            idx = registry.create(
                req.name, dimension=req.dimension,
                metric=req.metric, compression=compression,
            )
            s = idx.describe_index_stats()
        except FileExistsError as e:
            raise _http_error(409, str(e), "ALREADY_EXISTS")
        return {
            "name": req.name,
            "dimension": req.dimension,
            "metric": req.metric,
            "compression": s["compression"],
            "status": "ready",
        }

    @app.get("/indexes")
    async def list_indexes() -> dict:
        return {"indexes": registry.list_indexes()}

    @app.get("/indexes/{name}")
    async def describe_index(name: str) -> dict:
        from .errors import NotFound as _Nf
        for entry in registry.list_indexes():
            if entry["name"] == name:
                return entry
        raise _not_found(f"index {name!r} not found")

    @app.delete("/indexes/{name}")
    async def delete_index(name: str) -> dict:
        registry.delete(name)
        return {"deleted": name, "status": "deleted"}

    # ---- Auth middleware -----------------------------------------------
    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if auth.is_unauth():
            return await call_next(request)
        header = request.headers.get("Api-Key")
        if not header or not auth.check(header):
            return JSONResponse(
                status_code=401,
                content={"error": {"code": "UNAUTHENTICATED",
                                   "message": "invalid or missing Api-Key"}},
            )
        return await call_next(request)

    return app


def serve(data_root: str | Path, *, host: str = "127.0.0.1", port: int = 8080,
          api_key: str | None = None, log_level: str = "info") -> None:
    """Launch the uvicorn server. Blocks until shutdown."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "REST server requires uvicorn. Install with "
            "`pip install \"spectraltm-db[server]\"`."
        )
    app = build_app(data_root, api_key=api_key)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
