from compute_matrices import gaussian_1s_sto_3g

def generate_basis(structure, basis_type=gaussian_1s_sto_3g):
    import numpy as np
    basis = [basis_type(R, zeta=zeta) for _, R, _, zeta in structure]
    return basis

def solve_fock_equation(H_core, G, X_can):
    import numpy as np
    F = H_core + G
    F_dash = X_can.T @ F @ X_can
    epsilon, C_dash = np.linalg.eig(F_dash)
    epsilon = np.diag(epsilon)
    C = X_can @ C_dash
    return C, epsilon, F


def create_density_matrix_from_coefficients(C):
    number_of_occupied_orbitals = int((C.shape[0]) / 2)
    P = 2 * C[:, :number_of_occupied_orbitals].reshape((-1, 1)) @ C[:, :number_of_occupied_orbitals].reshape((1, -1))
    return P


def compute_G_matrix(H_core, P, two_electron_integrals):
    import numpy as np
    G = np.zeros(H_core.shape)
    for u in range(H_core.shape[0]):
        for v in range(H_core.shape[1]):
            temp_term = 0
            for w in range(two_electron_integrals.shape[2]):
                for x in range(two_electron_integrals.shape[3]):
                    temp_term += P[w, x] * (
                                two_electron_integrals[u, v, w, x] - (0.5 * two_electron_integrals[u, x, w, v]))
            G[u, v] = temp_term
    return G


def compute_total_energy(P, H_core, F):
    import numpy as np
    E = 0.5 * (np.sum(P.T * (H_core + F)))
    return E
