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
