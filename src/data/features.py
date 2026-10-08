"""Stage features: join VAA, EROM, StreamCat, USGS attributes, D50 and the aridity index.

Sources: protocol sections 2.3 (site_table), 2.7 (no imputation) and 3.5 (L4 banned list).
Writes reports/feature_missing.json.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
