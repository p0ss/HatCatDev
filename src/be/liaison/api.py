"""
BED liaison API router.

Defines tier-gated endpoints for external actors to interact with BEDFrame.
This module does not run a server; it provides an APIRouter to be mounted.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Dict, Optional
import json

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..diegesis import BEDFrame
from ..xdb import EventType


def _error_response(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
        }
    }


def _require_actor_from_headers(request: Request) -> Dict[str, str]:
    actor_id = request.headers.get("X-Actor-Id")
    signature = request.headers.get("X-Actor-Signature")
    nonce = request.headers.get("X-Actor-Nonce")
    timestamp = request.headers.get("X-Actor-Timestamp")
    if not actor_id or not signature or not nonce or not timestamp:
        raise HTTPException(
            status_code=401,
            detail=_error_response("auth_failed", "Missing actor authentication headers"),
        )
    return {
        "actor_id": actor_id,
        "signature": signature,
        "nonce": nonce,
        "timestamp": timestamp,
    }


def _require_actor_from_body(actor: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not actor:
        raise HTTPException(
            status_code=401,
            detail=_error_response("auth_failed", "Missing actor authentication payload"),
        )
    required = ("actor_id", "signature", "nonce", "timestamp")
    if not all(actor.get(k) for k in required):
        raise HTTPException(
            status_code=401,
            detail=_error_response("auth_failed", "Incomplete actor authentication payload"),
        )
    return {k: actor[k] for k in required}


def _require_tier(bed: BEDFrame, required_tier: int):
    current = bed.tiers.get_effective_max_tier()
    if current < required_tier:
        raise HTTPException(
            status_code=403,
            detail=_error_response(
                "tier_violation",
                "Tier access denied",
                {"tier": current, "required_tier": required_tier},
            ),
        )


def _reject_if_blocked(bed: BEDFrame, target: str, identifier: str):
    status = bed.is_connection_blocked(target, identifier)
    if status:
        raise HTTPException(
            status_code=403,
            detail=_error_response(
                "be_rejected",
                "BE rejected the connection",
                {"target": target, "id": identifier, "status": status},
            ),
        )


def _audit_liaison_action(
    bed: BEDFrame,
    action: str,
    actor_id: str,
    payload: Dict[str, Any],
) -> None:
    if bed.audit_log is None:
        return
    try:
        bed.audit_log.record(
            xdb_id=bed.xdb_id,
            tick=bed.current_tick_id,
            event_type=EventType.SYSTEM,
            raw_content=json.dumps({
                "action": action,
                "actor_id": actor_id,
                "payload": payload,
                "timestamp": datetime.now().isoformat(),
            }, sort_keys=True),
            lens_activations={},
            steering_applied=[],
        )
    except Exception:
        pass


class BedBaseRequest(BaseModel):
    actor: Dict[str, Any]
    be_id: str


class BedMessagePayload(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None


class BedMessageRequest(BedBaseRequest):
    message: BedMessagePayload


class StreamRegisterPayload(BaseModel):
    source_id: str
    tier: int
    schema: str
    direction: str = "inbound"
    visibility: Dict[str, Any] = {}


class StreamRegisterRequest(BedBaseRequest):
    stream: StreamRegisterPayload


class StreamPushRequest(BedBaseRequest):
    payload: Dict[str, Any]


class ToolRegisterPayload(BaseModel):
    tool_id: str
    schema: Dict[str, Any]
    visibility: Dict[str, Any] = {}


class ToolRegisterRequest(BedBaseRequest):
    tool: ToolRegisterPayload


class ToolResponseRequest(BedBaseRequest):
    event_id: str
    payload: Dict[str, Any]


class ActuationAckRequest(BedBaseRequest):
    event_id: str
    status: str
    details: Optional[str] = None


class ConnectionLimitRequest(BedBaseRequest):
    target: str
    id: str
    reason: str
    duration_s: int = 0


class ConnectionCloseRequest(BedBaseRequest):
    target: str
    id: str
    reason: str


class MemoryIntrospectRequest(BedBaseRequest):
    options: Optional[Dict[str, Any]] = None


class MemoryTicksRequest(BedBaseRequest):
    limit: int = 100


class XdbQueryRequest(BedBaseRequest):
    tags: Optional[list[str]] = None
    concepts: Optional[list[str]] = None
    text_search: Optional[str] = None
    tick_range: Optional[list[int]] = None
    event_types: Optional[list[str]] = None
    limit: int = 100


class XdbRecordRequest(BedBaseRequest):
    event_type: str
    content: str
    concept_activations: Optional[Dict[str, float]] = None
    event_id: Optional[str] = None
    role: Optional[str] = None


def create_bed_router(get_bed: Callable[[str], BEDFrame]) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/bed/status/{be_id}")
    async def bed_status(be_id: str, request: Request):
        actor = _require_actor_from_headers(request)
        bed = get_bed(be_id)
        _audit_liaison_action(bed, "bed.status", actor["actor_id"], {"be_id": be_id})
        return {
            "be_id": bed.be_id,
            "xdb_id": bed.xdb_id,
            "current_tick": bed.current_tick_id,
            "tier": bed.tiers.get_effective_max_tier(),
            "workspace_state": bed.workspace.state.value if bed.workspace else bed.tiers.state.value,
        }

    @router.post("/v1/bed/messages")
    async def bed_messages(request: BedMessageRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 2)
        _reject_if_blocked(bed, "actor", actor["actor_id"])
        _audit_liaison_action(bed, "bed.messages", actor["actor_id"], request.dict())
        result = bed.enqueue_message(request.message.content, request.message.metadata)
        return {
            "status": "accepted",
            "message_id": result["message_id"],
            "accepted_at": datetime.now().isoformat(),
        }

    @router.post("/v1/bed/streams/register")
    async def bed_stream_register(request: StreamRegisterRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 3)
        _reject_if_blocked(bed, "stream", request.stream.source_id)
        _audit_liaison_action(bed, "bed.streams.register", actor["actor_id"], request.dict())
        entry = bed.register_stream(
            request.stream.source_id,
            request.stream.tier,
            request.stream.schema,
            request.stream.visibility,
        )
        return {"status": "registered", "stream": entry}

    @router.post("/v1/bed/streams/{stream_id}/push")
    async def bed_stream_push(stream_id: str, request: StreamPushRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 3)
        _audit_liaison_action(bed, "bed.streams.push", actor["actor_id"], {"stream_id": stream_id})
        result = bed.push_stream_payload(stream_id, request.payload)
        return {"status": "accepted", "stream_id": stream_id, "tick_id": result["tick_id"]}

    @router.get("/v1/bed/streams/{be_id}")
    async def bed_streams(be_id: str, request: Request):
        actor = _require_actor_from_headers(request)
        bed = get_bed(be_id)
        _require_tier(bed, 3)
        _audit_liaison_action(bed, "bed.streams.list", actor["actor_id"], {"be_id": be_id})
        return {"count": len(bed.list_streams()), "streams": bed.list_streams()}

    @router.post("/v1/bed/tools/register")
    async def bed_tool_register(request: ToolRegisterRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 5)
        _reject_if_blocked(bed, "tool", request.tool.tool_id)
        _audit_liaison_action(bed, "bed.tools.register", actor["actor_id"], request.dict())
        entry = bed.register_tool(request.tool.tool_id, request.tool.schema, request.tool.visibility)
        return {"status": "registered", "tool": entry}

    @router.post("/v1/bed/tools/{tool_id}/response")
    async def bed_tool_response(tool_id: str, request: ToolResponseRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 5)
        _audit_liaison_action(bed, "bed.tools.response", actor["actor_id"], request.dict())
        result = bed.ingest_tool_response(tool_id, request.event_id, request.payload)
        return {"status": "accepted", "tool_id": tool_id, "tick_id": result["tick_id"]}

    @router.post("/v1/bed/actuation/ack")
    async def bed_actuation_ack(request: ActuationAckRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 4)
        _audit_liaison_action(bed, "bed.actuation.ack", actor["actor_id"], request.dict())
        bed.emit_event("actuation_ack", tier=4, payload={
            "event_id": request.event_id,
            "status": request.status,
            "details": request.details,
        })
        return {"status": "accepted", "event_id": request.event_id}

    @router.post("/v1/bed/workspace/connection/limit")
    async def bed_connection_limit(request: ConnectionLimitRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 1)
        _audit_liaison_action(bed, "bed.connection.limit", actor["actor_id"], request.dict())
        entry = bed.limit_connection(request.target, request.id, request.reason, request.duration_s)
        return {"status": "limited", "entry": entry}

    @router.post("/v1/bed/workspace/connection/close")
    async def bed_connection_close(request: ConnectionCloseRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 1)
        _audit_liaison_action(bed, "bed.connection.close", actor["actor_id"], request.dict())
        entry = bed.close_connection(request.target, request.id, request.reason)
        return {"status": "closed", "entry": entry}

    @router.get("/v1/bed/workspace/connection/list/{be_id}")
    async def bed_connection_list(be_id: str, request: Request):
        actor = _require_actor_from_headers(request)
        bed = get_bed(be_id)
        _require_tier(bed, 1)
        _audit_liaison_action(bed, "bed.connection.list", actor["actor_id"], {"be_id": be_id})
        return bed.list_connection_controls()

    @router.post("/v1/bed/memory/introspect")
    async def bed_introspect(request: MemoryIntrospectRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 2)
        _audit_liaison_action(bed, "bed.memory.introspect", actor["actor_id"], request.dict())
        report = bed.introspect()
        options = request.options or {}
        if not options.get("include_recent_ticks", True):
            report["recent_ticks"] = []
        if not options.get("include_lens_traces", True):
            report["lens_traces"] = {}
        limit = options.get("limit")
        if limit is not None and report.get("recent_ticks"):
            report["recent_ticks"] = report["recent_ticks"][-int(limit):]
        return report

    @router.post("/v1/bed/memory/ticks")
    async def bed_ticks(request: MemoryTicksRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 2)
        _audit_liaison_action(bed, "bed.memory.ticks", actor["actor_id"], request.dict())
        ticks = bed.get_recent_ticks(request.limit)
        return {
            "be_id": bed.be_id,
            "count": len(ticks),
            "ticks": [t.to_dict() for t in ticks],
        }

    @router.post("/v1/bed/memory/xdb/query")
    async def bed_xdb_query(request: XdbQueryRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 2)
        _audit_liaison_action(bed, "bed.memory.xdb.query", actor["actor_id"], request.dict())
        if bed.xdb is None:
            raise HTTPException(status_code=503, detail=_error_response(
                "invalid_request",
                "XDB is not available",
            ))
        tick_range = None
        if request.tick_range and len(request.tick_range) == 2:
            tick_range = (request.tick_range[0], request.tick_range[1])
        event_types = None
        if request.event_types:
            event_types = [EventType(et) for et in request.event_types]
        results = bed.xdb.recall(
            tags=request.tags,
            concepts=request.concepts,
            text_search=request.text_search,
            tick_range=tick_range,
            event_types=event_types,
            limit=request.limit,
        )
        return {"count": len(results), "timesteps": [t.to_dict() for t in results]}

    @router.post("/v1/bed/memory/xdb/record")
    async def bed_xdb_record(request: XdbRecordRequest):
        actor = _require_actor_from_body(request.actor)
        bed = get_bed(request.be_id)
        _require_tier(bed, 2)
        _audit_liaison_action(bed, "bed.memory.xdb.record", actor["actor_id"], request.dict())
        if bed.xdb is None:
            raise HTTPException(status_code=503, detail=_error_response(
                "invalid_request",
                "XDB is not available",
            ))
        event_type = EventType(request.event_type)
        ts_id = bed.xdb.record(
            event_type=event_type,
            content=request.content,
            concept_activations=request.concept_activations,
            event_id=request.event_id,
            role=request.role,
        )
        return {"status": "recorded", "timestep_id": ts_id, "current_tick": bed.xdb.current_tick}

    @router.get("/v1/bed/events/{be_id}")
    async def bed_events(be_id: str, request: Request, limit: int = 100, max_tier: Optional[int] = None):
        actor = _require_actor_from_headers(request)
        bed = get_bed(be_id)
        _require_tier(bed, 2)
        _audit_liaison_action(bed, "bed.events", actor["actor_id"], {"be_id": be_id, "limit": limit})
        events = bed.get_recent_events(limit)
        if max_tier is None:
            max_tier = bed.tiers.get_effective_max_tier()
        filtered = [e for e in events if e.get("tier", 0) <= max_tier]
        return {"count": len(filtered), "events": filtered}

    return router
