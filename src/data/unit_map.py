"""Stage unit_map: sentinels to missing, then the exact unit map U to SI.

Sources: protocol section 2.1; proofs section 1.2 (U), Propositions 1.1-1.3.
Reads params domain.s_floor.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
