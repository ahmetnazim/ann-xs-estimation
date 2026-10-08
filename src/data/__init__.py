"""Data ETL: one module per dvc.yaml stage, plus the raw-unit schemas.

Sources: protocol sections 1.5 (stages), 2 (schemas, QC) and 3.3 (fold table).
Only the split stage creates folds; all later code reads the frozen fold table.
Owning phase: 5. Placeholder (P4-D8 (a)); no numerical, physics or ML logic yet.
"""
