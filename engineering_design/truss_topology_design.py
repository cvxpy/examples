"""
Truss topology design finds the stiffest (or the lightest) structure made of bars
within a fixed "ground structure" of candidate members. Even though the nodal
displacements depend on the bar stiffnesses, the design problem is convex when
formulated with the compliance: minimizing the weight of a truss whose compliance
is bounded is a semidefinite program (Lobo, Vandenberghe, Boyd and Lebret,
"Applications of second-order cone programming", Linear Algebra and its
Applications 284:193-228, 1998).

A truss with n bars is described by cross-sectional areas a >= 0. For bar i of
length l_i connecting two nodes, the global stiffness matrix is

    K(a) = sum_i (a_i / l_i) * g_i g_i^T,

where g_i is the bar's (signed) direction vector in the free nodal degrees of
freedom. K(a) is linear in the design vector a, and the compliance
f^T K(a)^+ f of the structure under an external force f is bounded through the
Schur complement constraint

    [[K(a), f], [f^T, gamma]] >> 0.

Minimizing the weight sum_i l_i a_i subject to this linear matrix inequality
is an SDP. Bars whose optimal area is (near) zero drop out of the structure,
so the optimal topology emerges from the ground structure.
"""

import matplotlib.pyplot as plt
import numpy as np

import cvxpy as cp


def build_ground_structure(
    grid_shape: tuple[int, int],
    spacing: float = 1.0,
    max_length: float = 1.5,
    supported: tuple[int, ...] = (0, 1),
    loaded: int = 6,
    force: tuple[float, float] = (0.0, -1.0),
):
    """Build a 2D ground structure: candidate bars between nearby grid nodes.

    Returns the bar lengths, reduced direction vectors g_i (one per bar, in the
    free degrees of freedom only) and the external force vector.
    """
    cols, rows = grid_shape
    nodes = np.array(
        [(i * spacing, j * spacing) for i in range(cols) for j in range(rows)]
    )
    num_dofs = 2 * len(nodes)

    # Candidate bars connect every pair of nodes closer than max_length.
    pairs = [
        (p, q)
        for p in range(len(nodes))
        for q in range(p + 1, len(nodes))
        if np.linalg.norm(nodes[p] - nodes[q]) <= max_length
    ]
    lengths = np.array([np.linalg.norm(nodes[p] - nodes[q]) for p, q in pairs])

    # Fixed degrees of freedom (both coordinates of each supported node).
    fixed = np.sort(np.concatenate([[2 * n, 2 * n + 1] for n in supported]))
    free = np.setdiff1d(np.arange(num_dofs), fixed)

    directions = []
    for p, q in pairs:
        g = np.zeros(num_dofs)
        unit = (nodes[q] - nodes[p]) / lengths[len(directions)]
        g[2 * p : 2 * p + 2] = unit
        g[2 * q : 2 * q + 2] = -unit
        directions.append(g[free])

    f = np.zeros(num_dofs)
    f[2 * loaded : 2 * loaded + 2] = force

    return lengths, np.array(directions), f[free], nodes, pairs


def stiffness_matrix(areas, lengths, directions):
    """The global stiffness matrix K(a), linear in the bar areas."""
    return cp.sum([cp.outer(g * (a / l), g) for a, l, g in zip(areas, lengths, directions)])


def solve_truss_design(lengths, directions, f, gamma, a_max=1.0, a_min=1e-4):
    """Minimize the weight of a truss subject to a compliance limit.

    The Schur complement constraint

        [[K(a), f], [f^T, gamma]] >> 0

    encodes that the compliance f^T K(a)^+ f does not exceed gamma. The small
    lower bound on the areas keeps K(a) positive definite for the solver.
    """
    n = len(lengths)
    a = cp.Variable(n, name="areas")
    num_dofs = directions.shape[1]

    compliance = cp.bmat(
        [
            [stiffness_matrix(a, lengths, directions), cp.reshape(f, (num_dofs, 1), order="C")],
            [cp.reshape(f, (1, num_dofs), order="C"), cp.reshape(gamma, (1, 1), order="C")],
        ]
    )

    problem = cp.Problem(
        cp.Minimize(lengths @ a),
        [compliance >> 0, a >= a_min, a <= a_max],
    )
    # The optimal areas pile up near the compliance boundary, so the interior-point
    # solver needs tight tolerances to converge to an accurate solution.
    problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert problem.status == cp.OPTIMAL
    return a.value


def plot_truss(nodes, pairs, areas, tol=0.1):
    """Draw the surviving bars, with line width proportional to the area."""
    for (p, q), a in zip(pairs, areas):
        if a > tol:
            x = [nodes[p][0], nodes[q][0]]
            y = [nodes[p][1], nodes[q][1]]
            plt.plot(x, y, "o-", lw=8 * a, color="steelblue")
    plt.axis("equal")
    plt.axis("off")
    plt.title("Optimal topology")


if __name__ == "__main__":
    # A cantilever: the left edge is clamped, the load pulls the lower right corner down.
    lengths, directions, f, nodes, pairs = build_ground_structure(
        grid_shape=(4, 2),
        supported=(0, 1),
        loaded=6,
        force=(0.0, -1.0),
    )
    num_dofs = directions.shape[1]

    # Compliance of the fully solid ground structure (all bars at their maximum area),
    # which sets the scale for meaningful compliance limits.
    k_solid = np.zeros((num_dofs, num_dofs))
    for length, g in zip(lengths, directions):
        k_solid += np.outer(g, g) / length
    displacement, *_ = np.linalg.lstsq(k_solid, f, rcond=None)
    c_solid = float(f @ displacement)
    print(f"compliance of the fully solid structure: {c_solid:.2f}\n")

    gammas = [2.0, 3.0, 5.0, 10.0, 20.0]  # multiples of the solid-structure compliance

    plt.figure(figsize=(10, 4))
    for k, multiple in enumerate(gammas):
        gamma = multiple * c_solid
        areas = solve_truss_design(lengths, directions, f, gamma)
        weight = float(lengths @ areas)
        active = int(np.sum(areas > 0.1))
        print(
            f"compliance <= {multiple:4.1f} x solid: weight = {weight:7.4f}, "
            f"{active:2d} of {len(areas)} bars used"
        )
        plt.subplot(1, len(gammas), k + 1)
        plot_truss(nodes, pairs, areas)
        plt.title(f"$\\gamma = {multiple:.0f}C_\\mathrm{{solid}}$, weight = {weight:.2f}", fontsize=10)
    plt.tight_layout()
    plt.show()
