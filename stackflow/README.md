# StackFlow

StackFlow is a modular Python inventory management and reporting system.

## Project Structure

- `stackflow/models.py`: Data models (`Item`, `Transaction`).
- `stackflow/storage.py`: JSON-based storage engine for persistence.
- `stackflow/inventory.py`: Inventory manager for stock operations (add, restock, deduct, discount).
- `stackflow/reports.py`: Analytics and reporting engine (valuation, category breakdowns).
- `tests/test_inventory.py`: Unit tests for inventory operations.
- `tests/test_reports.py`: Unit tests for reporting analytics.

## Running Tests

From inside this `stackflow` directory, run the test suite with `pytest`:

```bash
pytest
```

Or using Python's standard `unittest` module:

```bash
python3 -m unittest discover -s tests
```
