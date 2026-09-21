#!/usr/bin/env python3
"""Run-environment switches for Trilinos-backed regression decks."""

import os
import subprocess


def enable_trilinos_debug(tpetra=True):
    """Switch on MueLu, and optionally Tpetra, internal checks.

    Returns 1 if mpiexec drops the variables, which would leave the checks
    silent, else 0.
    """
    names = ["MUELU_DEBUG"] + (["TPETRA_DEBUG"] if tpetra else [])
    for name in names:
        os.environ[name] = "1"
    probe = "echo " + "".join("$" + name for name in names)
    seen = subprocess.run("mpiexec -n 1 sh -c '%s'" % probe, shell=True,
                          capture_output=True, text=True).stdout.strip()
    if seen == "1" * len(names):
        return 0
    print("Failure: mpiexec did not forward %s, so the checks stay silent."
          % ", ".join(names))
    return 1
