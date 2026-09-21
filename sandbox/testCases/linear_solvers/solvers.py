"""Deck catalog: name -> (Solver YAML block, XML files to copy).

Three deck shapes, one builder each:

    ifpack2(...)         one preconditioner on the whole 2x2 system
    block_diagonal(...)  one per diagonal block, E-B coupling dropped
    blocktri(...)        pivot solve plus Schur complement

blocktri takes a pivot body and a Schur body. Build those with diagonal(),
direct(), amg(), or auxspace(). Bodies are written unindented and indented
once on the way in, so nesting stays readable here.

Block 0 is E (HCURL, edges), block 1 is B (HDIV, faces). Every blocktri deck
pivots on block 1, so the pivot is the HDIV mass matrix and the Schur
complement lands on E.
"""

AMG_PIVOT = "amg_pivot.xml"
CHEB = "cheb_1level.xml"
CHEB_MASS = "cheb_mass_1level.xml"

DAMPED_JACOBI = ("'relaxation: type': Jacobi\n"
                 "'relaxation: sweeps': 2\n"
                 "'relaxation: damping factor': 0.5\n")

DEGREE2_CHEB = ("'chebyshev: degree': 2\n"
                "'chebyshev: ratio eigenvalue': 7.0\n"
                "'chebyshev: eigenvalue max iterations': 15\n")


def _indent(body, n):
    if not body:
        return ""
    pad = " " * n
    return "".join(pad + line + "\n" for line in body.rstrip("\n").split("\n"))


# --- block bodies ------------------------------------------------------------

def diagonal(lumped=False):
    """Pivot only. Lumped is a signed row sum, not a sum of magnitudes."""
    body = "preconditioner type: Diagonal\n"
    if lumped:
        body += "diag use lumped diagonal: true\n"
    return body


def direct():
    """Amesos2 KLU2. A reference ceiling, far too slow for production."""
    return "preconditioner type: Direct\n"


def amg(xml):
    return "preconditioner type: AMG\nAMG Settings:\n  xml param file: %s\n" % xml


def auxspace(kind, xml, extra=""):
    """RefMaxwell or Maxwell1 on the edge space."""
    return ("preconditioner type: {k}\n"
            "hgrad basis name: phi_aux\n"
            "hcurl basis name: E\n"
            "{k} Settings:\n"
            "  xml param file: {x}\n"
            "{e}").format(k=kind, x=xml, e=extra)


# --- deck shapes -------------------------------------------------------------

def blocktri(pivot, schur, iters=500, approx="diag", lumped_weight=True,
             damping=None, triangle=None):
    head = "    max linear iters: %d\n" % iters
    if damping is not None:
        head += "    Schur damping: %s\n" % damping
    opts = ("      pivot block: 1\n"
            "      approximation type: %s\n"
            "      diag use lumped pivot diagonal: %s\n"
            % (approx, "true" if lumped_weight else "false"))
    if triangle is not None:
        opts += "      triangle: %s\n" % triangle
    return (head +
            "    preconditioner type: block triangular\n"
            "    Pivot Block Settings:\n" + _indent(pivot, 6) +
            "    Schur Block Settings:\n" + opts + _indent(schur, 6))


def refmaxwell(xml, extra="", pivot=None, triangle=None):
    """blocktri with RefMaxwell on the Schur block (addon disabled)."""
    return blocktri(amg(AMG_PIVOT) if pivot is None else pivot,
                    auxspace("RefMaxwell", xml,
                             extra),
                    iters=200, triangle=triangle)


def ifpack2(variant, params):
    """Monolithic. variant goes straight to Ifpack2::Factory, so any name it
    knows works: RELAXATION, CHEBYSHEV, RILUK, ILUT, SCHWARZ, ..."""
    return ("    max linear iters: 500\n"
            "    preconditioner type: Ifpack2\n"
            "    preconditioner variant: %s\n"
            "    Preconditioner Settings:\n" % variant) + _indent(params, 6)


def block_diagonal(variant, params=""):
    """Same smoother on both diagonal blocks, E-B coupling dropped."""
    body = "      preconditioner variant: %s\n" % variant + _indent(params, 6)
    return ("    max linear iters: 500\n"
            "    preconditioner type: block diagonal\n"
            "    Block 0 Settings:\n" + body +
            "    Block 1 Settings:\n" + body)


SOLVERS = {
    # --- baselines and the original ladder ---
    "jacobi": (ifpack2("RELAXATION", DAMPED_JACOBI), []),
    "block_diag": (block_diagonal("RELAXATION", DAMPED_JACOBI), []),
    "blocktri_jac_jac": (blocktri(diagonal(), amg("jacobi_1level.xml")),
                         [AMG_PIVOT, "jacobi_1level.xml"]),
    "blocktri_cheb": (blocktri(diagonal(), amg(CHEB)), [AMG_PIVOT, CHEB]),
    "blocktri_amg": (blocktri(amg(AMG_PIVOT), amg(AMG_PIVOT)), [AMG_PIVOT]),

    # --- RefMaxwell and Maxwell1 on the Schur block ---
    "refmaxwell_p2v3": (refmaxwell("refmaxwell_p2v3.xml"),
                        [AMG_PIVOT, "refmaxwell_p2v3.xml"]),
    "refmaxwell_p2v3_filter": (
        refmaxwell("refmaxwell_p2v3.xml", '  "filter SM": true\n'),
        [AMG_PIVOT, "refmaxwell_p2v3.xml"]),
    "refmaxwell_p2v3_filter_off": (
        refmaxwell("refmaxwell_p2v3.xml", '  "filter SM": false\n'),
        [AMG_PIVOT, "refmaxwell_p2v3.xml"]),
    "refmaxwell_dl015": (refmaxwell("refmaxwell_dl015.xml"),
                         [AMG_PIVOT, "refmaxwell_dl015.xml"]),
    "refmaxwell_dl02_mode1": (refmaxwell("refmaxwell_dl02_mode1.xml"),
                              [AMG_PIVOT, "refmaxwell_dl02_mode1.xml"]),
    "refmaxwell_dl02_additive": (refmaxwell("refmaxwell_dl02_additive.xml"),
                                 [AMG_PIVOT, "refmaxwell_dl02_additive.xml"]),
    "refmaxwell_dl02_121": (refmaxwell("refmaxwell_dl02_121.xml"),
                            [AMG_PIVOT, "refmaxwell_dl02_121.xml"]),
    "maxwell1_emin0": (blocktri(amg(AMG_PIVOT),
                                auxspace("Maxwell1", "maxwell1_emin0.xml"),
                                iters=200),
                       [AMG_PIVOT, "maxwell1_emin0.xml"]),

    # --- ILU on the Schur block; dies above CFL 2 ---
    "blocktri_ilu0": (blocktri(diagonal(), amg("ilu0_1level.xml")),
                      [AMG_PIVOT, "ilu0_1level.xml"]),
    "blocktri_ilu1": (blocktri(diagonal(), amg("ilu1_1level.xml")),
                      [AMG_PIVOT, "ilu1_1level.xml"]),
    "blocktri_amg_ilu": (blocktri(diagonal(), amg("amg_ilu.xml")),
                         [AMG_PIVOT, "amg_ilu.xml"]),
    "block_diag_ilu": (block_diagonal("RILUK"), []),

    # --- inverting the HDIV mass pivot; each changes one thing vs blocktri_cheb ---
    "blocktri_chebpivot": (blocktri(amg(CHEB_MASS), amg(CHEB)),
                           [AMG_PIVOT, CHEB, CHEB_MASS]),
    "blocktri_cheb_lumpedpivot": (blocktri(diagonal(lumped=True), amg(CHEB)),
                                  [AMG_PIVOT, CHEB]),
    "blocktri_cheb_pointweight": (blocktri(diagonal(), amg(CHEB),
                                           lumped_weight=False),
                                  [AMG_PIVOT, CHEB]),
    "blocktri_directpivot": (blocktri(direct(), amg(CHEB)), [AMG_PIVOT, CHEB]),
    # refmaxwell_p2v3 with a mass-tuned Chebyshev pivot instead of 10-level SA-AMG.
    "refmaxwell_p2v3_chebpivot": (
        refmaxwell("refmaxwell_p2v3.xml", pivot=amg(CHEB_MASS)),
        [AMG_PIVOT, "refmaxwell_p2v3.xml", CHEB_MASS]),

    # --- Schur approximation quality ---
    "blocktri_cheb_base": (blocktri(diagonal(), amg(CHEB), approx="base"),
                           [AMG_PIVOT, CHEB]),
    "blocktri_cheb_gamma05": (blocktri(diagonal(), amg(CHEB), damping=0.5),
                              [AMG_PIVOT, CHEB]),
    "blocktri_cheb_gamma08": (blocktri(diagonal(), amg(CHEB), damping=0.8),
                              [AMG_PIVOT, CHEB]),
    "blocktri_directschur": (blocktri(diagonal(), direct()), [AMG_PIVOT]),

    # --- triangle ordering; upper is the default via right preconditioner ---
    "blocktri_cheb_lower": (blocktri(diagonal(), amg(CHEB), triangle="lower"),
                            [AMG_PIVOT, CHEB]),
    "refmaxwell_p2v3_lower": (
        refmaxwell("refmaxwell_p2v3.xml", triangle="lower"),
        [AMG_PIVOT, "refmaxwell_p2v3.xml"]),

    # --- Gauss-Seidel and Schwarz ---
    "sgs": (ifpack2("RELAXATION", "'relaxation: type': Symmetric Gauss-Seidel\n"
                                  "'relaxation: sweeps': 1\n"
                                  "'relaxation: damping factor': 1.0\n"), []),
    # Ifpack2 Gauss-Seidel is only Gauss-Seidel within a rank, Jacobi across;
    # l1 is the variant meant for parallel.
    "sgs_l1": (ifpack2("RELAXATION", "'relaxation: type': Symmetric Gauss-Seidel\n"
                                     "'relaxation: sweeps': 1\n"
                                     "'relaxation: damping factor': 1.0\n"
                                     "'relaxation: use l1': true\n"
                                     "'relaxation: l1 eta': 1.5\n"), []),
    # jacobi's sweeps and damping, only the relaxation type changed.
    "sgs_damped": (ifpack2("RELAXATION",
                           "'relaxation: type': Symmetric Gauss-Seidel\n"
                           "'relaxation: sweeps': 2\n"
                           "'relaxation: damping factor': 0.5\n"), []),
    # Ifpack2 defaults for the subdomain solves.
    "schwarz": ("    max linear iters: 500\n"
                "    preconditioner type: domain decomposition\n", []),

    # --- one Chebyshev, three block shapes ---
    "cheb_mono": (ifpack2("CHEBYSHEV", DEGREE2_CHEB), []),
    "cheb_block_diag": (block_diagonal("CHEBYSHEV", DEGREE2_CHEB), []),
    "cheb_block_tri": (blocktri(amg(CHEB), amg(CHEB)), [AMG_PIVOT, CHEB]),
}
