def get_metadata(path,smi):
    """
    A function to extract the dataset size related  parameters and the numpy array required for ML.
    :param path: path to db (str)
    :param smi: smiles string (smi)
    :return: observables (floats)
    """
    import os, json

    smidict  = json.load(open('smi_file.json'))
    currentjson  = json.load(open(path+smidict[smi]))
    etot = currentjson['gaussian']['properties']['total_energy'] 
    dipole = currentjson['gaussian']['properties']['electric_dipole_moment_norm'] 
    quadrapole = currentjson['gaussian']['properties']['electric_quadrupole_moment_norm']
    solv = currentjson['gaussian']['properties']['SMD_solvation_energy']
    mp = currentjson['Tm']
 
    return etot, dipole, quadrapole, solv, mp



def get_inputdata(path):
    """
    A function to extract the smiles, MP dataset required for unsupervised learning.
    :param path: full path (str)
    :return: SMI (list), MP (list)
    """
    import os, json
    
    json_list = list()
    for filename in os.listdir(path):
        if 'json' in filename:
            json_list.append(filename)

    SMI_list = list()
    MP_list = []
    for item in json_list:
        sample = json.load(open(path+item))
        smi = sample['smiles']
        mp = float(sample['Tm'])
        MP_list.append(mp)
        SMI_list.append(smi)
    
    return  SMI_list, MP_list



def gen_numpy(path,smilist):
    """
    A function to generate numpy array for ML training/ validation
    :param path: full path to db (str)
    :param smiles list: smiles (list)
    Return type: numpy array 
    """
    import numpy as np
    from get_metadata import get_metadata
    from descriptor import Morgan2D
    nb = 1024
    trainlen = len(smilist)
    train_XY = np.zeros((trainlen, nb + 5))
    local_X = np.zeros((nb + 5))
    for num, smi in enumerate(smilist):
        
        etot, dipole, quadrapole, solv, mp = get_metadata(path,smi)
        ecfp = Morgan2D(smi,2,nb)
        local_X[0:nb] = ecfp
        local_X[-5] = etot
        local_X[-4] = dipole
        local_X[-3] = quadrapole
        local_X[-2] = solv
        local_X[-1] = mp 
        train_XY[num, :] = local_X
        
    return train_XY
