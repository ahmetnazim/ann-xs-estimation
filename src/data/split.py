"""Stage split: site-COMID groups and the frozen fold-assignment table.

Sources: protocol sections 3.1 (groups), 3.2 (families), 3.3 (table). Reads params seeds, splits.
The ONLY module allowed to create folds; its SHA-256 enters the protocol by ADL in Phase 5.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
