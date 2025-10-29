def gaussian_1s_sto_3g(R, coefficients=[0.444635,0.535328,0.154329], exponents=[0.109,0.405,2.227], zeta=1):
    import numpy as np
    R = np.array(R)
#     exponents = list(np.round(np.array(exponents)*(zeta**2),3))
    exponents = list(np.array(exponents)*(zeta**2))
    return R, coefficients, exponents


def integrate_two_primitve_gaussian(alpha,beta,R_A,R_B):
    import numpy as np
    p = alpha+beta
    R_A = np.array(R_A)
    R_B = np.array(R_B)
    diff_norm = np.linalg.norm(R_A-R_B)
#     integral = ((2*alpha*beta/(p*np.pi))**(3/4))*((np.pi/p)**(3/2))*np.exp((-1*alpha*beta/p)*(diff_norm)**2)
    integral = (np.exp((-1*alpha*beta/p)*(diff_norm)**2))
    integral *= (p**(-3/2))
    integral *= ((2*alpha)**(3/4))*((2*beta)**(3/4)) ## normalization of gaussian integrals
    return integral


def kinetic_energy_integral_two_primitve_gaussian(alpha,beta,R_A,R_B):
    import numpy as np
    p = alpha+beta
    R_A = np.array(R_A)
    R_B = np.array(R_B)
    diff_norm = np.linalg.norm(R_A-R_B)
    integral = (np.exp((-1*alpha*beta/p)*(diff_norm)**2))
    integral *= (p**(-3/2))
    integral *= (alpha*beta)/p
    integral *= ((2*alpha)**(3/4))*((2*beta)**(3/4)) ## normalization of gaussian integrals
    integral *= (3 - (2*(alpha*beta)*((diff_norm)**2)/p))
    return integral


def evaluate_F_0(t):
    import math
    import numpy as np
    if t < 1e-8:
        # Taylor expansion: F0(0)=1 - t/3 + ...
        return 1.0 - t/3.0
    else:
        return 0.5*(math.sqrt(np.pi/t))*math.erf(t**0.5)


def nuclear_attracton_integral_two_primitve_gaussian(alpha,beta,R_A,R_B,Z_C,R_C):
    import numpy as np
    p = alpha+beta
    R_A = np.array(R_A)
    R_B = np.array(R_B)
    R_P = (alpha*R_A + beta*R_B)/p
#     print(R_P-R_C)
    diff_norm = np.linalg.norm(R_A-R_B)
    diff_norm_C = np.linalg.norm(R_P-R_C)
    integral = (np.exp((-1*alpha*beta/p)*((diff_norm)**2)))
    integral *= -1*(2*np.pi/p)*Z_C
    t = p*(diff_norm_C**2)
    F_0 = evaluate_F_0(t)
    integral *= F_0
    integral *= ((2*alpha/np.pi)**(3/4))*((2*beta/np.pi)**(3/4)) ## normalization of gaussian integrals
    return integral


def two_e_integral_four_primitve_gaussian(alpha, beta, gamma, delta, R_A, R_B, R_C, R_D):
    import numpy as np
    p = alpha + beta
    q = gamma + delta
    R_A = np.array(R_A)
    R_B = np.array(R_B)
    R_C = np.array(R_C)
    R_D = np.array(R_D)
    R_P = (alpha * R_A + beta * R_B) / p
    R_Q = (gamma * R_C + delta * R_D) / q

    diff_norm_P = np.linalg.norm(R_A - R_B)
    diff_norm_Q = np.linalg.norm(R_C - R_D)
    diff_norm_PQ = np.linalg.norm(R_P - R_Q)

    integral = np.exp((-1 * alpha * beta / p) * ((diff_norm_P) ** 2) + (-1 * gamma * delta / q) * ((diff_norm_Q) ** 2))
    integral *= 2 * (np.pi ** (2.5)) / (p * q * np.sqrt(p + q))
    t = p * q * (diff_norm_PQ ** 2) / (p + q)
    F_0 = evaluate_F_0(t)
    integral *= F_0
    integral *= ((2 * alpha / np.pi) ** (3 / 4)) * (
                (2 * beta / np.pi) ** (3 / 4))  ## normalization of gaussian integrals
    integral *= ((2 * gamma / np.pi) ** (3 / 4)) * (
                (2 * delta / np.pi) ** (3 / 4))  ## normalization of gaussian integrals
    return integral
