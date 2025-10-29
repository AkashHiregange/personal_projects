from compute_integrals import *

def gaussian_1s_sto_3g(R, coefficients=[0.444635,0.535328,0.154329], exponents=[0.109,0.405,2.227], zeta=1):
    import numpy as np
    R = np.array(R)
#     exponents = list(np.round(np.array(exponents)*(zeta**2),3))
    exponents = list(np.array(exponents)*(zeta**2))
    return R, coefficients, exponents

def overlap_matrix(basis_set):
    import numpy as np
    mat_size = len(basis_set)
    S = np.zeros((mat_size,mat_size))
    for u in range(len(basis_set)):
        for v in range(len(basis_set)):
            if u==v:
                overlap_mat_element=0
                for i in range(len(basis_set[u][2])):
                    for j in range(len(basis_set[v][2])):
                        alpha =basis_set[u][2][i]
                        beta = basis_set[v][2][j]
                        R_A = basis_set[u][0]
                        R_B = basis_set[v][0]
                        integral = basis_set[u][1][i]*basis_set[v][1][j]
                        integral *= integrate_two_primitve_gaussian(alpha, beta, R_A, R_B)
                        overlap_mat_element+=integral
                S[u,v] = np.round(overlap_mat_element,5)
            elif u>v:
                overlap_mat_element=0
                for i in range(len(basis_set[u][2])):
                    for j in range(len(basis_set[v][2])):
                        alpha =basis_set[u][2][i]
                        beta = basis_set[v][2][j]
                        R_A = basis_set[u][0]
                        R_B = basis_set[v][0]
                        integral = basis_set[u][1][i]*basis_set[v][1][j]
                        integral *= integrate_two_primitve_gaussian(alpha, beta, R_A, R_B)
                        overlap_mat_element+=integral
                S[u,v] = np.round(overlap_mat_element,5)
                S[v,u] = np.round(overlap_mat_element,5)
    return S

def h_core(basis_set, structure):
    import numpy as np
    mat_size = len(basis_set)
    V = np.zeros((mat_size,mat_size))
    T = np.zeros((mat_size,mat_size))
    for ind, info in enumerate(structure):
        Z_C = info[2]
        R_C = info[1]
        for u in range(len(basis_set)):
            for v in range(len(basis_set)):
                nuclear_attraction_mat_element=0
                ke_mat_element=0
                for i in range(len(basis_set[u][2])):
                    for j in range(len(basis_set[v][2])):
                        alpha =basis_set[u][2][i]
                        beta = basis_set[v][2][j]
                        R_A = basis_set[u][0]
                        R_B = basis_set[v][0]
                        nuc_att_integral = basis_set[u][1][i]*basis_set[v][1][j]
                        nuc_att_integral *= nuclear_attracton_integral_two_primitve_gaussian(alpha, beta, R_A, R_B, Z_C, R_C)
                        nuclear_attraction_mat_element+=nuc_att_integral
                        if ind == len(structure)-1:
                            integral = basis_set[u][1][i]*basis_set[v][1][j]
                            integral *= kinetic_energy_integral_two_primitve_gaussian(alpha, beta, R_A, R_B)
                            ke_mat_element+=integral
                V[u,v] += np.round(nuclear_attraction_mat_element,5)
                if ind == len(structure)-1:
                    T[u,v] = np.round(ke_mat_element,5)
    return T, V


def two_electron_integral_matrix(basis_set):
    import numpy as np
    I = np.zeros((2,2,2,2))
    for u in range(len(basis_set)):
        for v in range(len(basis_set)):
            for w in range(len(basis_set)):
                for x in range(len(basis_set)):
                    two_e_integral_mat_element=0
                    for i in range(len(basis_set[u][2])):
                        for j in range(len(basis_set[v][2])):
                            for k in range(len(basis_set[w][2])):
                                for l in range(len(basis_set[x][2])):
                                    alpha =basis_set[u][2][i]
                                    beta = basis_set[v][2][j]
                                    gamma = basis_set[w][2][k]
                                    delta = basis_set[x][2][l]
                                    R_A = basis_set[u][0]
                                    R_B = basis_set[v][0]
                                    R_C = basis_set[w][0]
                                    R_D = basis_set[x][0]
                                    integral = basis_set[u][1][i]*basis_set[v][1][j]
                                    integral *= basis_set[w][1][k]*basis_set[x][1][l]
                                    integral *= two_e_integral_four_primitve_gaussian(alpha, beta, gamma, delta,
                                                                                     R_A, R_B, R_C, R_D)
                                    two_e_integral_mat_element+=integral
                    I[u,v,w,x] = np.round(two_e_integral_mat_element,5)
    return I


def create_unitary_from_overlap(overlap_matrix):
    import numpy as np
    U = np.linalg.eig(overlap_matrix)[1]
    return U


def create_s_half(overlap_matrix, unitary_transformation):
    import numpy as np
    s = np.round(unitary_transformation.T @ overlap_matrix @ unitary_transformation, 4)
    s_diagonal = 1 / np.sqrt(np.diagonal(s))
    s_half = np.diag(s_diagonal)
    return s_half


def canonical_transformation(unitary_transformation, s_half):
    import numpy as np
    X_can = unitary_transformation @ s_half
    return X_can
