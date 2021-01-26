import rdkit
import rdkit.Chem as Chem
import numpy as np

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']


atom_feature_dimension = len(elem_list) + 6 + 6 + 6 + 1  # Number of atom features is 82.
bond_feature_dimension = 6  # Number of bond features is 6.
max_nb = 10  # Max number of neighbours for each atom is 10.

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()], dtype=np.float32)

def smiles2graph(smiles, idxfunc=lambda x:x.GetIdx()):
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

    for bond in mol.GetBonds():
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        if number_neighbours[a1] == max_nb or number_neighbours[a2] == max_nb:
            raise Exception(smiles)
        atom_neighbours[a1,number_neighbours[a1]] = a2
        atom_neighbours[a2,number_neighbours[a2]] = a1
        bond_neighbours[a1,number_neighbours[a1]] = idx
        bond_neighbours[a2,number_neighbours[a2]] = idx
        number_neighbours[a1] += 1
        number_neighbours[a2] += 1
        feature_bonds[idx] = bond_features(bond)
    return feature_atoms, feature_bonds, atom_neighbours, bond_neighbours, number_neighbours

def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def pack2D_withidx(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M, 2))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m,0] = i
        a[i,0:n,0:m,1] = arr
    return a

def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i,0:n] = arr
    return a

def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        for j in range(arr.shape[0]):
            a[i][j] = 1
    return a

def smiles2graph_list(smiles_list, idxfunc=lambda x:x.GetIdx()):
    '''
    This function prepares all of the model inputs needed to process one batch and
    pads them as needed (because not all examples will have the same number of atoms)
    '''

    # smiles2graph function for one molecule. smiles2graph_list function for a batch of molecules.

    res = list(map(lambda x:smiles2graph(x,idxfunc), smiles_list))
    fatom_list, fbond_list, gatom_list, gbond_list, nb_list = zip(*res)
    return pack2D(fatom_list), pack2D(fbond_list), pack2D_withidx(gatom_list), pack2D_withidx(gbond_list), pack1D(nb_list), get_mask(fatom_list)

if __name__ == "__main__":
    np.set_printoptions(threshold='nan')
    a,b,c,d,e,f = smiles2graph_list(["c1cccnc1",
                                     # 'c1nccc2n1ccc2'
                                     ])
    print("Atom features for a batch of molecules:")
    print(a.shape)  # (1,6,82) Batch size is 1. Atom number is 6. Atom feature is 82.
    print(a)
    print("Bond features for a batch of molecules:")
    print(b.shape)  # (1,6,6) Batch size is 1. Bond number is 6. Bond feature is 6.
    print(b)

    print("gatom_list:")
    print(c.shape)  # (1, 6, 10, 2) Atom number is 6. Max neighbour is 10.
    print(c)

    print("gbond_list:")
    print(d.shape)  # (1, 6, 10, 2) Bond number is 6. Max neighbour is 10.
    print(d)

    print("Neighbours(bond number) of each atom:")
    print(e.shape)  # (1, 6) Bond number is 6.
    print(e)
    print(f)
