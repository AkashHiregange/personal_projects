import numpy as np
from scf_cycle import perform_rhf_scf_calculation
from compute_matrices import gaussian_1s_sto_3g

structure = [['He', [1.4632,0,0], 2, 2.09], ['H', [0,0,0], 1, 1.24]]

perform_rhf_scf_calculation(structure=structure, basis_type=gaussian_1s_sto_3g,scf_accuracy_energy=1e-7)
