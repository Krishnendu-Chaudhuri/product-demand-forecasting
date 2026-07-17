"""Backward-compatible shim for pipeline CLI."""

from pipelines.cli import _prepare_hydra_argv, main

if __name__ == "__main__":
    _prepare_hydra_argv()
    main()
