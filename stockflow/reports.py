"""Analytics and reporting engine for inventory."""

from typing import Dict, List

from stockflow.inventory import InventoryManager
from stockflow.models import Item


class InventoryReporter:
    """Generates analytical reports from inventory data."""

    def __init__(self, manager: InventoryManager):
        self.manager = manager

    def total_inventory_value(self) -> float:
        """Calculates total valuation of all stock on hand (sum of price * quantity)."""
        items = self.manager.list_all_items()
        return round(sum(item.price * item.quantity for item in items), 2)

    def get_low_stock_items(self, threshold: int = 5) -> List[Item]:
        """Returns items where quantity is at or below the given threshold."""
        return [
            item for item in self.manager.list_all_items() if item.quantity <= threshold
        ]

    def generate_category_summary(self) -> Dict[str, float]:
        """
        Calculates total inventory valuation grouped by category.
        Returns a dictionary mapping category name to total value of all items in that category.
        """
        summary: Dict[str, float] = {}
        for item in self.manager.list_all_items():
            item_total = item.price * item.quantity
            summary[item.category] = round(item_total, 2)
        return summary
