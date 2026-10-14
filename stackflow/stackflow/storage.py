"""Storage layer for persisting inventory data."""

import json
from pathlib import Path
from typing import Dict, Optional

from stackflow.models import Item


class JsonStorage:
    """Manages reading and writing inventory items to a JSON file."""

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = Path(file_path) if file_path else None

    def load(self) -> Dict[str, Item]:
        """Loads inventory items from disk. Returns empty dict if file does not exist."""
        if not self.file_path or not self.file_path.exists():
            return {}

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {sku: Item.from_dict(item_data) for sku, item_data in data.items()}

    def save(self, items: Dict[str, Item]) -> None:
        """Saves current inventory items to disk."""
        if not self.file_path:
            return

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = {sku: item.to_dict() for sku, item in items.items()}

        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
