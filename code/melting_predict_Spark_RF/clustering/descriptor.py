def Morgan2D(smi,radius=2,nb=1024):
    """
    A function to create Morgan ECFP
    :param smi: smiles (str)
    :param radius: test set split fraction (float)
    :param nb: number of bits(int)
    Return type: ecfp  (binary list)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nb) 
    return ecfp


