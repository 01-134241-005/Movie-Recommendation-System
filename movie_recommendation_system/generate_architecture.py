"""Generate the professional architecture.svg diagram for the project report."""

from __future__ import annotations

from utils import BASE_DIR


def main() -> None:
    """Confirm that the maintained SVG diagram is available."""
    architecture_path = BASE_DIR / "architecture.svg"
    if not architecture_path.exists():
        raise FileNotFoundError("architecture.svg is missing from the project root.")
    print(f"Architecture diagram ready: {architecture_path}")


if __name__ == "__main__":
    main()
