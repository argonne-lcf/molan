def gen_dists(smileslist,nb=1024):
    """
    A function to generate Tanimoto distance matrix
    :param smileslist: smiles (list)
    :param nb: number of bits (int)
    :return: dists (Tanimoto distance matrix), molobj (list),nfps (int)
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    molobj = [Chem.MolFromSmiles(smi) for smi in smileslist]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nb) for mol in molobj]
    
    
    from rdkit import DataStructs
    # first generate the distance matrix:
    dists = []
    nfps = len(fps)
    for i in range(1,nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        dists.extend([1-x for x in sims])
    return dists, molobj, nfps



def gen_fineclusters(dists,nfps,cutoff=0.2):
    """
    A function to generate fine grained clusters (i.e. Butina) from Tanimoto distance matrices
    :param dists: Tanimoto distance matrix
    :param nfps: number of fingerprints
    :param cutoff: radial cutoff (float)
    :return: Butinacluster
    """
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Butina
    
    # now cluster the data:
    Butinacluster = Butina.ClusterData(dists,nfps,cutoff,isDistData=True, reordering=True)
    return Butinacluster



def gen_coarseclusters(dists,nfps):
    """
    A function to generate coarse grained clusters (i.e. Murtagh) from Tanimoto distance matrices
    :param dists: Tanimoto distance matrix
    :param nfps: number of fingerprints
    :return: cs (clusters)
    """
    from rdkit import DataStructs
    from rdkit.ML.Cluster import Murtagh
    
    # now cluster the data:
    cs = Murtagh.ClusterData(dists,nfps,Murtagh.WARDS,isDistData=1)
    return cs



def postprocess_coarsecluster(Murtaghcluster,numcluster,smilist,mplist):
    """
    A function to output a chosen number of hierarchical coarse grained clusters (i.e. Murtagh)
    :param Murtaghcluster:
    :param numcluster: number of cluster to output
    :param smilist: (list)
    :param mplist: (list)
    :return: rawclustdict (dict), clustdict (dict)
    """
    from rdkit.ML.Cluster import ClusterUtils
    
    splitClusts = ClusterUtils.SplitIntoNClusters(Murtaghcluster[0],numcluster)
    #centroids = [ClusterUtils.FindClusterCentroidFromDists(x,dists) for x in splitClusts]
    rawclustdict = dict()
    clustdict = dict()
    #clustdict['centroids'] = centroids
    
    for index, cluster in enumerate(splitClusts):
        children = cluster.GetPoints()
        pts = [x.GetData() for x in children]
        rawclustdict[index+1] = pts
        clustdict[index+1] = list()
        
        for pt in pts:
            clustdict[index+1].append({'index': pt , 'smiles': smilist[pt],'MP': mplist[pt]})
    
    return rawclustdict, clustdict



def writejson(filename,jsondata):
    import json
    with open(filename+ '.json', 'w') as f:
        json.dump(jsondata, f,indent=4)



def write_coarsecluster(Murtaghcluster,Nclustlist,smilist,mplist,outfile):
    """
    A function to output a list of  hierarchical coarse grained clusters (i.e. Murtagh) to json
    :param Murtaghcluster:
    :param Nclustlist: list of number of cluster to output
    :param smilist: (list)
    :param mplist: (list)
    :param outfile : output json file name (str)
    :return: json
    """
    from copy import deepcopy
    from gen_cluster import postprocess_coarsecluster
    from gen_cluster import writejson

    template = {'algorithm': 'Murtagh',
    'method': 'Wards',
        'DB': [{'All': [{
                        'clusters': [],
                        'size': len(smilist) }],
               } ] }
    writedict = deepcopy(template)
    for numcluster in Nclustlist:
        print("Generating datasets for {} cluster size".format(numcluster))
        Allrawdict, Allclustdict = postprocess_coarsecluster(All_m_clusters,numcluster, All_smi, All_mp)
        writedict['DB'][0]['All'][0]['clusters'].append({'numcluster': len(Allclustdict), 'val': Allclustdict})

    writejson(outfile,writedict)
    return writedict



def write_finecluster(Butinacluster,smilist,mplist,outfile):
    """
        A function to output fine grained clusters (i.e. Butina) to json
        :param Butinacluster: (tuple)
        :param smilist: (list)
        :param mplist: (list)
        :param outfile : output json file name (str)
        :return: json
        """
    from copy import deepcopy
    from gen_cluster import writejson
    
    template = {'algorithm': 'Butina',
        'DB': [{'All': [{
                        'clusters': [],
                        'size': len(smilist) }],
               } ] }
    writedict = deepcopy(template)
    clustdict = dict()
    for count, tuples in enumerate(Butinacluster):
        clustdict[count+1] = list()
        for index in tuples:
            clustdict[count+1].append({'index': index , 'smiles': smilist[index],'MP': mplist[index]})


    writedict['DB'][0]['All'][0]['clusters'].append({'numcluster': len(clustdict), 'val': clustdict})
    
    writejson(outfile,writedict)
    return writedict
