"""ETL: verified evidence -> normalized artifacts -> DuckDB warehouse."""

from __future__ import annotations

from dd_lab.etl.warehouse import Warehouse, open_warehouse

__all__ = ["Warehouse", "open_warehouse"]
