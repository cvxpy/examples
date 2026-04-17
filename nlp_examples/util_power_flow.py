import numpy as np
from scipy.sparse import csr_matrix, diags

def create_admittance_matrices():
    N = 9
    # Branch data: (from_bus, to_bus, resistance, reactance, susceptance)
    branch_data = np.array([
        [0, 3, 0.0, 0.0576, 0.0],
        [3, 4, 0.017, 0.092, 0.158],
        [5, 4, 0.039, 0.17, 0.358],
        [2, 5, 0.0, 0.0586, 0.0],
        [5, 6, 0.0119, 0.1008, 0.209],
        [7, 6, 0.0085, 0.072, 0.149],
        [1, 7, 0.0, 0.0625, 0.0],
        [7, 8, 0.032, 0.161, 0.306],
        [3, 8, 0.01, 0.085, 0.176],
    ])

    M = branch_data.shape[0]  # Number of branches
    base_MVA = 100

    # Build incidence matrix A
    from_bus = branch_data[:, 0].astype(int)
    to_bus = branch_data[:, 1].astype(int)
    A = csr_matrix((np.ones(M), (from_bus, np.arange(M))), shape=(N, M)) + \
        csr_matrix((-np.ones(M), (to_bus, np.arange(M))), shape=(N, M))

    # Network impedance
    z = (branch_data[:, 2] + 1j * branch_data[:, 3]) / base_MVA

    # Bus admittance matrix Y_0
    Y_0 = A @ diags(1.0 / z) @ A.T

    # Shunt admittance from line charging
    y_sh = 0.5 * (1j * branch_data[:, 4]) * base_MVA
    Y_sh_diag = np.array((A @ diags(y_sh) @ A.T).diagonal()).flatten()
    Y_sh = diags(Y_sh_diag)

    # Extract conductance and susceptance matrices
    G0 = np.real(Y_0.toarray())  # Conductance matrix
    B0 = np.imag(Y_0.toarray())  # Susceptance matrix
    G_sh = np.real(Y_sh.toarray())  # Shunt conductance
    B_sh = np.imag(Y_sh.toarray())  #

    return G0, B0, G_sh, B_sh


