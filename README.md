# StockFlow

StockFlow is a modular Python inventory management and reporting system.

## Project Structure

- `stockflow/models.py`: Data models (`Item`, `Transaction`).
- `stockflow/storage.py`: JSON-based storage engine for persistence.
- `stockflow/inventory.py`: Inventory manager for stock operations (add, restock, deduct, discount).
- `stockflow/reports.py`: Analytics and reporting engine (valuation, category breakdowns).
- `tests/test_inventory.py`: Unit tests for inventory operations.
- `tests/test_reports.py`: Unit tests for reporting analytics.

## Running Tests

Run the test suite with `pytest`:

```bash
pytest
```

Or using Python's standard `unittest` module:

```bash
python3 -m unittest discover -s tests
```
