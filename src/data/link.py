"""Stage link: site_no to NHDPlusV2.1 COMID via GageLoc, snap fallback, drainage-area check.

Sources: protocol section 2.5; ADL-011. Reads params link.
Writes reports/link_counts.json, including the GageLoc vs k-NN disagreement rate.
Owning phase: 5. Placeholder (P4-D8 (a)): main() raises until Phase 5, so
a premature `dvc repro` or CLI call fails loudly instead of writing empty outputs.
"""


def main() -> None:
    """Run this entry point; not implemented until Phase 5."""
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    main()
