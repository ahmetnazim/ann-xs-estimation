"""Stage ind_prepare: eHydro independent set, deduplicated against training reaches.

Sources: protocol sections 1.1 (ehydro, O-5 (c)), 3.2 (S-IND) and 3.5 (L8).
Reads params splits.ind_dedupe_m.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
