"""VillageSQL column types used by the ORM."""

import math
import re
from datetime import datetime, timezone

from sqlalchemy.dialects.mysql import DATETIME as MySQLDateTime
from sqlalchemy.types import TypeDecorator, UserDefinedType

# Digits, sign, decimal point, and exponent only — safe to inline as a SQL string.
_VECTOR_LITERAL = re.compile(r"\[[-+0-9.eE,]+\]")
_NUMBER = re.compile(r"[-+0-9.eE]+")


def vector_literal(values: list[float] | tuple[float, ...] | str) -> str:
    """Render a vector as a vsql_vector literal, e.g. ``[0.1,0.2]``.

    The server accepts this form as a SQL string constant. Server-side
    placeholders inside distance functions are rejected, so callers that put
    the literal into ``COSINE_DISTANCE`` must inline this string.
    """
    if isinstance(values, str):
        literal = values.replace(" ", "")
        if not _VECTOR_LITERAL.fullmatch(literal):
            raise ValueError("vector literal contains unexpected characters")
        return literal

    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("embedding must be a non-empty sequence of finite floats")

    parts: list[str] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("embedding values must be finite")
        part = format(number, ".8g")
        if not _NUMBER.fullmatch(part):
            raise ValueError(f"unexpected numeric format: {part}")
        parts.append(part)
    return "[" + ",".join(parts) + "]"


class SVector(UserDefinedType):
    """Fixed-dimension ``SVECTOR(N)`` column (vsql_vector, stored in InnoDB)."""

    cache_ok = True

    def __init__(self, dim: int):
        self.dim = int(dim)

    def get_col_spec(self, **kw):
        return f"SVECTOR({self.dim})"

    def bind_processor(self, dialect):
        dim = self.dim

        def process(value):
            if value is None:
                return None
            if not isinstance(value, str) and len(value) != dim:
                raise ValueError(f"expected a {dim}-dimension vector, got {len(value)}")
            return vector_literal(value)

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None or isinstance(value, list):
                return value
            if isinstance(value, (bytes, bytearray)):
                value = value.decode()
            text = str(value).strip()
            if text == "[]":
                return []
            if not (text.startswith("[") and text.endswith("]")):
                return value
            return [float(part) for part in text[1:-1].split(",") if part.strip()]

        return process


class UTCDateTime(TypeDecorator):
    """Naive UTC ``DATETIME(6)`` in MySQL, timezone-aware ``datetime`` in Python."""

    impl = MySQLDateTime
    cache_ok = True

    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(MySQLDateTime(fsp=6))

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise TypeError("expected a datetime")
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
