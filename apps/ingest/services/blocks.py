"""A block is one heading or one paragraph, before chunking."""
from dataclasses import asdict, dataclass


@dataclass
class Block:
    text: str
    page: int | None = None
    is_heading: bool = False
    heading_level: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Block":
        return cls(
            text=data["text"],
            page=data.get("page"),
            is_heading=bool(data.get("is_heading", False)),
            heading_level=int(data.get("heading_level") or 0),
        )
