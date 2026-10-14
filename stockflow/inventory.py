"""Inventory management service."""

from typing import Dict, List, Optional

from stockflow.models import Item, Transaction
from stockflow.storage import JsonStorage


class ItemNotFoundError(KeyError):
    """Raised when an item SKU does not exist in inventory."""
    pass


class InsufficientStockError(ValueError):
    """Raised when attempting to deduct more items than available in stock."""
    pass


class InventoryManager:
    """Manages stock operations, item tracking, and state persistence."""

    def __init__(self, storage: Optional[JsonStorage] = None):
        self.storage = storage or JsonStorage()
        self._items: Dict[str, Item] = self.storage.load()
        self.history: List[Transaction] = []

    def add_item(self, sku: str, name: str, category: str, price: float, quantity: int = 0) -> Item:
        """Registers a new item in inventory."""
        if sku in self._items:
            raise ValueError(f"Item with SKU '{sku}' already exists.")
        if price < 0:
            raise ValueError("Price cannot be negative.")
        if quantity < 0:
            raise ValueError("Initial quantity cannot be negative.")

        item = Item(sku=sku, name=name, category=category, price=price, quantity=quantity)
        self._items[sku] = item
        self.history.append(Transaction(sku=sku, action="ADD", quantity_changed=quantity))
        self.storage.save(self._items)
        return item

    def get_item(self, sku: str) -> Item:
        """Retrieves an item by SKU or raises ItemNotFoundError."""
        if sku not in self._items:
            raise ItemNotFoundError(f"Item with SKU '{sku}' not found.")
        return self._items[sku]

    def list_all_items(self) -> List[Item]:
        """Returns all items currently in inventory."""
        return list(self._items.values())

    def restock(self, sku: str, quantity: int) -> Item:
        """Adds stock to an existing item."""
        if quantity <= 0:
            raise ValueError("Restock quantity must be positive.")
        item = self.get_item(sku)
        item.quantity += quantity
        self.history.append(Transaction(sku=sku, action="RESTOCK", quantity_changed=quantity))
        self.storage.save(self._items)
        return item

    def deduct_stock(self, sku: str, quantity: int) -> Item:
        """Reduces stock of an item when sold or removed."""
        if quantity <= 0:
            raise ValueError("Deduct quantity must be positive.")
        item = self.get_item(sku)
        if item.quantity < quantity:
            raise InsufficientStockError(
                f"Cannot deduct {quantity} units; only {item.quantity} available."
            )
        item.quantity -= quantity
        self.history.append(Transaction(sku=sku, action="DEDUCT", quantity_changed=-quantity))
        self.storage.save(self._items)
        return item

    def apply_discount(self, sku: str, discount_percent: float) -> Item:
        """
        Applies a percentage discount to an item's price.
        For example, applying a 20% discount to an item priced at $100.00 sets its new price to $80.00.
        """
        if not (0.0 <= discount_percent <= 100.0):
            raise ValueError("Discount percentage must be between 0.0 and 100.0.")
        item = self.get_item(sku)
        item.price = round(item.price * (discount_percent / 100.0), 2)
        self.history.append(
            Transaction(
                sku=sku,
                action="DISCOUNT",
                quantity_changed=0,
                notes=f"Discount {discount_percent}% applied, new price: {item.price}",
            )
        )
        self.storage.save(self._items)
        return item
