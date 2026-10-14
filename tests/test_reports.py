"""Tests for inventory reporting."""

import unittest

from stockflow.inventory import InventoryManager
from stockflow.reports import InventoryReporter


class TestInventoryReporter(unittest.TestCase):
    """Test suite for InventoryReporter."""

    def setUp(self):
        self.manager = InventoryManager()
        self.reporter = InventoryReporter(self.manager)

    def test_total_inventory_value(self):
        self.manager.add_item("SKU-1", "Widget A", "Tools", 10.0, 5)   # Value: 50.0
        self.manager.add_item("SKU-2", "Widget B", "Tools", 20.0, 3)   # Value: 60.0
        self.assertEqual(self.reporter.total_inventory_value(), 110.0)

    def test_get_low_stock_items(self):
        self.manager.add_item("SKU-1", "Pen", "Stationery", 2.0, 2)
        self.manager.add_item("SKU-2", "Notebook", "Stationery", 5.0, 15)
        low_stock = self.reporter.get_low_stock_items(threshold=5)
        self.assertEqual(len(low_stock), 1)
        self.assertEqual(low_stock[0].sku, "SKU-1")

    def test_generate_category_summary_aggregates_multiple_items(self):
        # Category "Electronics" has 2 products:
        # Monitor: 10 * 75.0 = 750.0
        # Mouse: 10 * 25.0 = 250.0
        # Total Electronics valuation should be 750.0 + 250.0 = 1000.0.
        self.manager.add_item("SKU-E1", "Monitor", "Electronics", 75.0, 10)
        self.manager.add_item("SKU-E2", "Mouse", "Electronics", 25.0, 10)
        # Category "Books" has 1 product:
        # Book: 5 * 20.0 = 100.0
        self.manager.add_item("SKU-B1", "Python Guide", "Books", 20.0, 5)

        summary = self.reporter.generate_category_summary()

        self.assertEqual(summary.get("Books"), 100.0)
        self.assertEqual(summary.get("Electronics"), 1000.0)

    def test_empty_inventory(self):
        self.assertEqual(self.reporter.total_inventory_value(), 0.0)
        self.assertEqual(self.reporter.get_low_stock_items(), [])
        self.assertEqual(self.reporter.generate_category_summary(), {})


if __name__ == "__main__":
    unittest.main()
