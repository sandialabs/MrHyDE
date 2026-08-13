# thermal_hess_vec_check_lq

Distributed-source thermal optimal control on $[0,1]^3$ with a known closed-form optimum.

## Problem

Steady Poisson, homogeneous Dirichlet. State $T$ is HGRAD-1, control $u$ (`ctrl_source`) is distributed HGRAD-1, regularizer weight $\gamma$.

$$
\begin{aligned}
\min_u \quad & J(u) = \tfrac12 \int_\Omega T^2 \, dx + \tfrac12 \gamma \int_\Omega u^2 \, dx \\
\text{s.t.} \quad & -\Delta T = 12\pi^2 \sin(2\pi x)\sin(2\pi y)\sin(2\pi z) + u, \\
& T = 0 \text{ on } \partial\Omega.
\end{aligned}
$$


## Run

```bash
./run.sh                  # default: check + exact/fd solves at N = 4, 8, 16
./run.sh check            # gradient + LQ-operator identity checks only
./run.sh exact-1e-4-N32   # single case: exact hessvec, gamma=1e-4, NX=32
NP=8 ./run.sh             # override MPI ranks (default 4)
```

Logs in `logs/mrhyde_<mode>.log`. `run.sh` prints the summary table on exit.

## Results ($\gamma = 10^{-4}$, np = 4, 2x2x1, Initial Radius $10^8$)

`./run.sh check` (exact HessVec, inner-solve floor):

| GRAD-CHECK | HESSVEC-CHECK | SECANT-IDENTITY | HV-BILINEARITY |
|-----------:|--------------:|----------------:|---------------:|
| 2.24e-07   | 1.03e-14      | 9.44e-15        | 8.22e-16       |

`./run.sh` (default sweep):

|mode|iter0 value|final value|final gnorm|native L2 err|n_outer|sum CG|wall (s)|
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|exact-1e-4-N4|4.17e-02|2.81e-02|1.43e-19|2.70e-02|**2**|44|4.2|
|fd-1e-4-N4|4.17e-02|2.81e-02|2.81e-19|2.70e-02|3|392|32.6|
|exact-1e-4-N8|5.64e-02|3.43e-02|8.68e-20|3.50e-03|**2**|93|9.4|
|fd-1e-4-N8|5.64e-02|3.43e-02|4.12e-19|3.50e-03|3|462|63.5|
|exact-1e-4-N16|6.09e-02|3.59e-02|1.26e-20|5.78e-04|**2**|147|26.2|
|fd-1e-4-N16|6.09e-02|3.59e-02|1.15e-19|5.78e-04|6|445|126.6|

## Exact vs FD HessVec

`MrHyDE_Objective::hessVec` switches on whether the inactive scalar `src_gate`
exists in Parameters:

- Present (`other_decks/exact/`): exact path. Wrap the physical source as
  `src_gate*(...)`, then `incrementalForwardModel` sets `src_gate=0` for the
  tangent, restores it, and `incrementalAdjointModel` assembles $Hv$.
- Absent (`other_decks/fd/`): falls back to `ROL::Objective::hessVec`
  (FD-of-gradients). That is the default if you do not add `src_gate`.

```yaml
# other_decks/exact/input_params.yaml + input_functions.yaml
src_gate:
  type: scalar
  value: 1.0
  usage: inactive
thermal source: 'src_gate*12.0*pi*pi*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z) + ctrl_source'
```

## Files

```text
input_solve.yaml.template     # solve deck; NELEM, GAMMA, OTHER_DECKS_DIR templated
run.sh                        # dispatcher + summary
rol_decks/rol_solve.yaml      # Trust Region + Truncated CG
other_decks/exact/            # ctrl_source + src_gate  -> exact HessVec
other_decks/fd/               # ctrl_source only        -> FD HessVec

# Operator checks (./run.sh check):
input_base.yaml               # check deck (N=8)
rol_decks/rol_check.yaml      # Do grad+hessvec / secant / bilinearity flags
```
