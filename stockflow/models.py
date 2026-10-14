"""Data models for items and transactions."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class Item:
    """Represents a product item in inventory."""

    sku: str
    name: str
    category: str
    price: float
    quantity: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Serializes Item to dictionary."""
        return {
            "sku": self.sku,
            "name": self.name,
            "category": self.category,
            "price": self.price,
            "quantity": self.quantity,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Item":
        """Deserializes Item from dictionary."""
        return cls(
            sku=data["sku"],
            name=data["name"],
            category=data["category"],
            price=float(data["price"]),
            quantity=int(data.get("quantity", 0)),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass
class Transaction:
    """Record of an inventory stock change."""

    sku: str
    action: str  # "ADD", "RESTOCK", "DEDUCT", "DISCOUNT"
    quantity_changed: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: Optional[str] = None
