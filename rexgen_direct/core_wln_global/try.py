import rdkit
import rdkit.Chem as Chem
import numpy as np
def smiles2graph(smiles, idxfunc=lambda x: x.GetIdx()):
    # This function is used for one molecule(smiles).

    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse smiles string:", smiles)

    number_atoms = mol.GetNumAtoms()
    # print("Number of atoms:")
    # print(number_atoms)

    number_bonds = max(mol.GetNumBonds(), 1)
    # print("Number of bonds:")
    # print(number_bonds)

    feature_atoms = np.zeros((number_atoms, atom_feature_dimension))
    feature_bonds = np.zeros((number_bonds, bond_feature_dimension))

    atom_neighbours = np.zeros((number_atoms, max_nb), dtype=np.int32)
    bond_neighbours = np.zeros((number_atoms, max_nb), dtype=np.int32)
    number_neighbours = np.zeros((number_atoms,), dtype=np.int32)

    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        if idx >= number_atoms:
            raise Exception(smiles)
        feature_atoms[idx] = atom_features(atom)