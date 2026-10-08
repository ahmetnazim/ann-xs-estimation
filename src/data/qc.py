"""Stage qc: row-wise, deterministic QC rules R1-R14 with flags retained.

Sources: protocol sections 2.2 (bounds) and 2.4 (rules). Reads params qc.
Writes reports/qc_counts.json with row and site counts per rule.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
