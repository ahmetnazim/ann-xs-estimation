"""Thin CLI: train one model on one family/fold from the frozen fold table.

Sources: protocol sections 3.3, 3.4 and 4.1-4.2.
Owning phase: 6. Placeholder (P4-D8 (a)): main() raises until Phase 6, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 6."""
    raise NotImplementedError("Phase 6")


if __name__ == "__main__":
    main()
