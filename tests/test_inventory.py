"""Tests for inventory management."""

import unittest

from stockflow.inventory import (
    InventoryManager,
    ItemNotFoundError,
    InsufficientStockError,
)


class TestInventoryManager(unittest.TestCase):
    """Test suite for InventoryManager."""

    def setUp(self):
        self.manager = InventoryManager()

    def test_add_item_success(self):
        item = self.manager.add_item("SKU-1", "Mechanical Keyboard", "Electronics", 75.0, 10)
        self.assertEqual(item.sku, "SKU-1")
        self.assertEqual(item.name, "Mechanical Keyboard")
        self.assertEqual(item.price, 75.0)
        self.assertEqual(item.quantity, 10)

    def test_add_duplicate_sku_raises_error(self):
        self.manager.add_item("SKU-1", "Item A", "General", 10.0, 5)
        with self.assertRaises(ValueError):
            self.manager.add_item("SKU-1", "Item B", "General", 15.0, 2)

    def test_get_nonexistent_item_raises_error(self):
        with self.assertRaises(ItemNotFoundError):
            self.manager.get_item("NONEXISTENT")

    def test_restock_increases_quantity(self):
        self.manager.add_item("SKU-2", "USB-C Cable", "Accessories", 12.0, 5)
        item = self.manager.restock("SKU-2", 15)
        self.assertEqual(item.quantity, 20)

    def test_deduct_stock_reduces_quantity(self):
        self.manager.add_item("SKU-3", "Gaming Mouse", "Electronics", 50.0, 10)
        item = self.manager.deduct_stock("SKU-3", 4)
        self.assertEqual(item.quantity, 6)

    def test_deduct_excessive_stock_raises_error(self):
        self.manager.add_item("SKU-4", "Desk Lamp", "Furniture", 30.0, 3)
        with self.assertRaises(InsufficientStockError):
            self.manager.deduct_stock("SKU-4", 5)

    def test_apply_discount(self):
        # A 20% discount on a $100.00 item should reduce the price to $80.00.
        self.manager.add_item("SKU-5", "Noise Canceling Headphones", "Electronics", 100.0, 8)
        updated_item = self.manager.apply_discount("SKU-5", 20.0)
        self.assertEqual(updated_item.price, 80.0)


if __name__ == "__main__":
    unittest.main()
