"""
Hand-written protobuf parser for the gateway_queries schema.

Provides the same interface as a protoc-generated pb2 file — specifically
ClientQueryProtobuf.FromString(bytes) and IndexerQueryProtobuf.FromString(bytes) —
without requiring protoc at build time or the protobuf package at runtime.

Schema source: gateway_queries.proto (transcribed from titorelli src/messages.rs).
"""

import struct
from typing import List, Optional


def _read_varint(buf: bytes, pos: int) -> tuple:
    """Read a varint from buf starting at pos. Returns (value, new_pos)."""
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            return result, pos


def _read_field(buf: bytes, pos: int) -> tuple:
    """Read one protobuf field tag + value. Returns (field_number, value, new_pos)."""
    tag, pos = _read_varint(buf, pos)
    field_number = tag >> 3
    wire_type = tag & 0x7

    if wire_type == 0:  # varint
        value, pos = _read_varint(buf, pos)
    elif wire_type == 1:  # 64-bit (double)
        (value,) = struct.unpack_from("<d", buf, pos)
        pos += 8
    elif wire_type == 2:  # length-delimited (string, bytes, embedded message)
        length, pos = _read_varint(buf, pos)
        value = buf[pos : pos + length]
        pos += length
    elif wire_type == 5:  # 32-bit (float)
        (value,) = struct.unpack_from("<f", buf, pos)
        pos += 4
    else:
        raise ValueError(f"Unknown wire type {wire_type} for field {field_number}")

    return field_number, value, pos


def _decode_str(value) -> str:
    """Decode a bytes value to str, replacing invalid UTF-8."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


# ---------------------------------------------------------------------------
# Optimized extraction functions (skip full object construction)
# ---------------------------------------------------------------------------


def _skip_field(buf: bytes, pos: int, wire_type: int) -> int:
    """Advance past a field value without decoding it."""
    if wire_type == 0:  # varint
        while buf[pos] & 0x80:
            pos += 1
        pos += 1
    elif wire_type == 1:  # 64-bit
        pos += 8
    elif wire_type == 2:  # length-delimited
        length, pos = _read_varint(buf, pos)
        pos += length
    elif wire_type == 5:  # 32-bit
        pos += 4
    else:
        raise ValueError(f"Unknown wire type {wire_type}")
    return pos


def _parse_submessage_keys(buf: bytes) -> tuple:
    """Extract (indexer_bytes, deployment_bytes, fee_grt) from an IndexerQueryProtobuf."""
    pos = 0
    n = len(buf)
    indexer = b""
    deployment = b""
    fee_grt = 0.0
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:  # indexer bytes
            length, pos = _read_varint(buf, pos)
            indexer = buf[pos : pos + length]
            pos += length
        elif field_number == 2 and wire_type == 2:  # deployment bytes
            length, pos = _read_varint(buf, pos)
            deployment = buf[pos : pos + length]
            pos += length
        elif field_number == 6 and wire_type == 1:  # fee_grt double
            (fee_grt,) = struct.unpack_from("<d", buf, pos)
            pos += 8
        else:
            pos = _skip_field(buf, pos, wire_type)
    return (indexer, deployment, fee_grt)


def extract_keys_and_fees(buf: bytes) -> List[tuple]:
    """
    Pass 1 minimal parser for ClientQueryProtobuf.

    Returns list of (indexer_bytes, deployment_bytes, fee_grt) tuples,
    one per indexer attempt. No string decoding or object allocation
    beyond tuples.
    """
    pos = 0
    n = len(buf)
    results: List[tuple] = []
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 10 and wire_type == 2:  # indexer_queries submessage
            length, pos = _read_varint(buf, pos)
            submsg_buf = buf[pos : pos + length]
            pos += length
            results.append(_parse_submessage_keys(submsg_buf))
        else:
            pos = _skip_field(buf, pos, wire_type)
    return results


def _parse_submessage_sample(buf: bytes) -> dict:
    """Extract sample fields from an IndexerQueryProtobuf submessage."""
    pos = 0
    n = len(buf)
    indexer_bytes = b""
    deployment_bytes = b""
    indexed_chain = ""
    url = ""
    fee_grt = 0.0
    response_time_ms = 0
    result = ""
    blocks_behind = 0
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 1 and wire_type == 2:  # indexer
            length, pos = _read_varint(buf, pos)
            indexer_bytes = buf[pos : pos + length]
            pos += length
        elif field_number == 2 and wire_type == 2:  # deployment
            length, pos = _read_varint(buf, pos)
            deployment_bytes = buf[pos : pos + length]
            pos += length
        elif field_number == 4 and wire_type == 2:  # indexed_chain
            length, pos = _read_varint(buf, pos)
            indexed_chain = buf[pos : pos + length].decode("utf-8", errors="replace")
            pos += length
        elif field_number == 5 and wire_type == 2:  # url
            length, pos = _read_varint(buf, pos)
            url = buf[pos : pos + length].decode("utf-8", errors="replace")
            pos += length
        elif field_number == 6 and wire_type == 1:  # fee_grt
            (fee_grt,) = struct.unpack_from("<d", buf, pos)
            pos += 8
        elif field_number == 7 and wire_type == 0:  # response_time_ms
            response_time_ms, pos = _read_varint(buf, pos)
        elif field_number == 9 and wire_type == 2:  # result
            length, pos = _read_varint(buf, pos)
            result = buf[pos : pos + length].decode("utf-8", errors="replace")
            pos += length
        elif field_number == 11 and wire_type == 0:  # blocks_behind
            blocks_behind, pos = _read_varint(buf, pos)
        else:
            pos = _skip_field(buf, pos, wire_type)
    return {
        "indexer_bytes": indexer_bytes,
        "deployment_bytes": deployment_bytes,
        "indexed_chain": indexed_chain,
        "url": url,
        "fee_grt": fee_grt,
        "response_time_ms": response_time_ms,
        "result": result,
        "blocks_behind": blocks_behind,
    }


def extract_sample_fields(buf: bytes) -> tuple:
    """
    Pass 2 selective parser for ClientQueryProtobuf.

    Returns (query_id, attempts_list) where each attempt is a dict with:
    indexer_bytes, deployment_bytes, indexed_chain, url, fee_grt,
    response_time_ms, result, blocks_behind.

    Skips 8 unused outer fields and 3 unused submessage fields
    (allocation, seconds_behind, indexer_errors).
    """
    pos = 0
    n = len(buf)
    query_id = ""
    attempts: List[dict] = []
    while pos < n:
        tag, pos = _read_varint(buf, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if field_number == 3 and wire_type == 2:  # query_id
            length, pos = _read_varint(buf, pos)
            query_id = buf[pos : pos + length].decode("utf-8", errors="replace")
            pos += length
        elif field_number == 10 and wire_type == 2:  # indexer_queries submessage
            length, pos = _read_varint(buf, pos)
            submsg_buf = buf[pos : pos + length]
            pos += length
            attempts.append(_parse_submessage_sample(submsg_buf))
        else:
            pos = _skip_field(buf, pos, wire_type)
    return (query_id, attempts)


class IndexerQueryProtobuf:
    """Parsed IndexerQueryProtobuf message (one per indexer attempt)."""

    __slots__ = [
        "indexer",
        "deployment",
        "allocation",
        "indexed_chain",
        "url",
        "fee_grt",
        "response_time_ms",
        "seconds_behind",
        "result",
        "indexer_errors",
        "blocks_behind",
    ]

    def __init__(self) -> None:
        self.indexer: bytes = b""
        self.deployment: bytes = b""
        self.allocation: bytes = b""
        self.indexed_chain: str = ""
        self.url: str = ""
        self.fee_grt: float = 0.0
        self.response_time_ms: int = 0
        self.seconds_behind: int = 0
        self.result: str = ""
        self.indexer_errors: str = ""
        self.blocks_behind: int = 0

    @classmethod
    def FromString(cls, buf: bytes) -> "IndexerQueryProtobuf":
        msg = cls()
        pos = 0
        n = len(buf)
        while pos < n:
            field_number, value, pos = _read_field(buf, pos)
            if field_number == 1:
                msg.indexer = value
            elif field_number == 2:
                msg.deployment = value
            elif field_number == 3:
                msg.allocation = value
            elif field_number == 4:
                msg.indexed_chain = _decode_str(value)
            elif field_number == 5:
                msg.url = _decode_str(value)
            elif field_number == 6:
                msg.fee_grt = value
            elif field_number == 7:
                msg.response_time_ms = value
            elif field_number == 8:
                msg.seconds_behind = value
            elif field_number == 9:
                msg.result = _decode_str(value)
            elif field_number == 10:
                msg.indexer_errors = _decode_str(value)
            elif field_number == 11:
                msg.blocks_behind = value
            # Unknown fields are silently skipped per protobuf spec.
        return msg


class ClientQueryProtobuf:
    """Parsed ClientQueryProtobuf message (one per gateway query)."""

    __slots__ = [
        "gateway_id",
        "receipt_signer",
        "query_id",
        "api_key",
        "result",
        "response_time_ms",
        "request_bytes",
        "response_bytes",
        "total_fees_usd",
        "indexer_queries",
        "user_id",
        "subgraph",
    ]

    def __init__(self) -> None:
        self.gateway_id: str = ""
        self.receipt_signer: bytes = b""
        self.query_id: str = ""
        self.api_key: str = ""
        self.result: str = ""
        self.response_time_ms: int = 0
        self.request_bytes: int = 0
        self.response_bytes: Optional[int] = None
        self.total_fees_usd: float = 0.0
        self.indexer_queries: List[IndexerQueryProtobuf] = []
        self.user_id: Optional[str] = None
        self.subgraph: Optional[str] = None

    @classmethod
    def FromString(cls, buf: bytes) -> "ClientQueryProtobuf":
        msg = cls()
        pos = 0
        n = len(buf)
        while pos < n:
            field_number, value, pos = _read_field(buf, pos)
            if field_number == 1:
                msg.gateway_id = _decode_str(value)
            elif field_number == 2:
                msg.receipt_signer = value
            elif field_number == 3:
                msg.query_id = _decode_str(value)
            elif field_number == 4:
                msg.api_key = _decode_str(value)
            elif field_number == 5:
                msg.result = _decode_str(value)
            elif field_number == 6:
                msg.response_time_ms = value
            elif field_number == 7:
                msg.request_bytes = value
            elif field_number == 8:
                msg.response_bytes = value
            elif field_number == 9:
                msg.total_fees_usd = value
            elif field_number == 10:
                if isinstance(value, bytes):
                    msg.indexer_queries.append(IndexerQueryProtobuf.FromString(value))
            elif field_number == 11:
                msg.user_id = _decode_str(value)
            elif field_number == 12:
                msg.subgraph = _decode_str(value)
            # Unknown fields are silently skipped per protobuf spec.
        return msg
