import rdkit.Chem as Chem
from .mol_graph import bond_feature_dimension, bond_features
import numpy as np

BOND_TYPE = ["NOBOND",
             Chem.rdchem.BondType.SINGLE,
             Chem.rdchem.BondType.DOUBLE,
             Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
N_BOND_CLASS = len(BOND_TYPE)
binary_feature_dimension = 4 + bond_feature_dimension
INVALID_BOND = -1

# in/out utlils

def get_binary_feature(reactant, max_natoms):
    '''
    This function is used to generate
    descriptions of atom-atom relationships,
    including the bond type between the atoms (if any) and whether they belong to the same molecule.
    It is used in the
    global attention mechanism.
    '''
    compounds = {}
    for index, string in enumerate(reactant.split('.')):
        mol = Chem.MolFromSmiles(string)
        for atom in mol.GetAtoms():
            compounds[atom.GetIntProp('molAtomMapNumber') - 1] = index
    number_compound = len(reactant.split('.'))
    rmol = Chem.MolFromSmiles(reactant)
    n_atoms = rmol.GetNumAtoms()
    bond_map = {}
    for bond in rmol.GetBonds():
        a1 = bond.GetBeginAtom().GetIntProp('molAtomMapNumber') - 1
        a2 = bond.GetEndAtom().GetIntProp('molAtomMapNumber') - 1
        bond_map[(a1, a2)] = bond_map[(a2,a1)] = bond
        
    binary_features = []
    for i in range(max_natoms):
        for j in range(max_natoms):

            binary_feature = np.zeros((binary_feature_dimension,))
            if i >= n_atoms or j >= n_atoms or i == j:
                binary_features.append(binary_feature)
                continue
            if (i, j) in bond_map:
                bond = bond_map[(i, j)]
                binary_feature[1:1 + bond_feature_dimension] = bond_features(bond)
            else:
                binary_feature[0] = 1.0

            #  binary_feature_dimension = bond_feature_dimension + 4
            binary_feature[-4] = 1.0 if compounds[i] != compounds[j] else 0.0
            binary_feature[-3] = 1.0 if compounds[i] == compounds[j] else 0.0
            binary_feature[-2] = 1.0 if number_compound == 1 else 0.0
            binary_feature[-1] = 1.0 if number_compound > 1 else 0.0
            binary_features.append(binary_feature)

    return np.vstack(binary_features).reshape((max_natoms, max_natoms, binary_feature_dimension))

bond_to_index  = {0.0: 0, 1:1, 2:2, 3:3, 1.5:4}
nbos = len(bond_to_index)
def get_bond_label(reactant, edits, max_natoms):
    rmol = Chem.MolFromSmiles(reactant)
    number_atoms = rmol.GetNumAtoms()
    rmap = np.zeros((max_natoms, max_natoms, nbos))
    
    #  edits:  1-2-0.0; 13-2-1.0
    for string in edits.split(';'):
        a1, a2, bond = string.split('-')
        x = min(int(a1)-1, int(a2)-1)
        y = max(int(a1)-1, int(a2)-1)
        z = bond_to_index[float(bond)]
        rmap[x,y,z] = rmap[y,x,z] = 1

    labels = []
    special_labels = []
    for i in range(max_natoms):
        for j in range(max_natoms):
            for k in range(len(bond_to_index)):
                if i == j or i >= number_atoms or j >= number_atoms:
                    labels.append(INVALID_BOND)  # mask
                else:
                    labels.append(rmap[i, j, k])
                    if rmap[i, j, k] == 1:
                        special_labels.append(i * max_natoms * nbos + j * nbos + k)
                        # TODO: check if this is consistent with how TF does flattening
    return np.array(labels), special_labels

def get_all_batch(re_list):
    mol_list = []
    max_natoms = 0
    for r,e in re_list:
        rmol = Chem.MolFromSmiles(r)
        mol_list.append((r,e))
        if rmol.GetNumAtoms() > max_natoms:
            max_natoms = rmol.GetNumAtoms()
    labels = []
    features = []
    sp_labels = []
    for r,e in mol_list:
        l, sl = get_bond_label(r,e,max_natoms)
        features.append(get_binary_feature(r, max_natoms))
        labels.append(l)
        sp_labels.append(sl)
    return np.array(features), np.array(labels), sp_labels

def get_binary_feature_batch(reactant_list):
    max_number_atoms = 0
    for reactant in reactant_list:
        rmol = Chem.MolFromSmiles(reactant)
        if rmol.GetNumAtoms() > max_number_atoms:
            max_number_atoms = rmol.GetNumAtoms()

    features = []
    for reactant in reactant_list:
        features.append(get_binary_feature(reactant, max_number_atoms))
    return np.array(features)
