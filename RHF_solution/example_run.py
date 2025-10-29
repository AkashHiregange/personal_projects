#IMPORTS
import numpy as np
from ase.build import bulk
from ase.phonons import Phonons
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from ase import Atoms
from ase.io import read, write
import numpy
import os
import shutil
from phonopy.interface.calculator import read_crystal_structure, write_crystal_structure



def make_matrix(atoms_object):
    num1 = int(input("What dimension would you like your supercell to be? "))
    num2 = float(input("What displacement would you like to use? "))

    atoms_object.write('geometry.in')
    unitcell, optional_structure_info = read_crystal_structure("geometry.in", interface_mode='aims')
    os.remove('geometry.in')

    sup_matrix = np.array([[num1, 0, 0], [0, num1, 0], [0, 0, num1]])
    det = np.round(np.linalg.det(sup_matrix))
    phonon = Phonopy(unitcell, supercell_matrix=sup_matrix)

    phonon.generate_displacements(distance=num2)
    supercells = phonon.supercells_with_displacements

    return det, supercells


def get_charges_and_moments(determinant, atoms_object):
    charges2 = []
    moments2 = []

    charges = atoms_object.get_initial_charges()
    moments = atoms_object.get_initial_magnetic_moments()

    for i in range(len(atoms_object)):
        for j in range(int(determinant)):
            charges2.append(charges[i])
            moments2.append(moments[i])

    return moments2, charges2


def creating_files_and_directories(atoms_object, charges, moments):

    chem_sym = atoms_object.get_chemical_symbols()
    for ind, sup in enumerate(supercells):
        write_crystal_structure(f"geometry_{ind+1:03}.in", supercells[ind], interface_mode='aims')
        atoms = read(f"geometry_{ind+1:03}.in")
        atoms.set_initial_charges(charges)
        atoms.set_initial_magnetic_moments(moments)
        directory = f"disp_{ind+1:03}"
        parent_dir = os.getcwd()
        path_final = os.path.join(parent_dir, directory)
        if os.path.exists(path_final):
            shutil.rmtree(path_final)
            os.mkdir(path_final)
            atoms.write(path_final + '/geometry.in')
            shutil.copy(parent_dir + '/input.py', path_final + '/input.py')
            shutil.copy(parent_dir + '/submission.script', path_final + '/submission.script')
            os.chdir(path_final)
            try:
                os.system("qsub submission.script")
            except:
                print(' ')
            os.chdir(parent_dir)
        else:
            os.mkdir(path_final)
            atoms.write(path_final + '/geometry.in')
            shutil.copy(parent_dir + '/input.py', path_final + '/input.py')
            shutil.copy(parent_dir + '/submission.script', path_final + '/submission.script')
            os.chdir(path_final)
            try:
                os.system("qsub submission.script")
            except:
                print(' ')
            os.chdir(parent_dir)
        os.remove(f"geometry_{ind+1:03}.in")


crys = bulk('Pt', 'fcc', a=4.121)

det, supercells = make_matrix(crys)
moments, charges = get_charges_and_moments(det,crys)

creating_files_and_directories(crys, charges, moments)

