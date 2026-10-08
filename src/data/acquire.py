"""Stage acquire@<unit>: fetch one manifest unit into data/raw/<unit> and verify it.

Sources: protocol sections 1.1 (manifest), 1.2 (checksums, immutability), 1.5; ADL-043.
Units: the 11 dvc.yaml foreach items. Reads params acquire. Writes MANIFEST.sha256;
fails on a checksum mismatch and never overwrites.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
