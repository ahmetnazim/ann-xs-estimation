"""Stage validate_raw: rules V1-V5 on every raw unit.

Sources: protocol section 2.3 (raw-unit schemas). Writes reports/validate_raw.json;
a unit fails only on V1, V2 or V3.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
