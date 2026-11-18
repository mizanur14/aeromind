"""AeroMind: uncertainty-aware turbofan RUL prediction on NASA C-MAPSS."""

__version__ = "0.3.0"


def main() -> int:
    """Console entry point (delegates to :func:`aeromind.cli.main`)."""
    from aeromind.cli import main as cli_main

    return cli_main()
