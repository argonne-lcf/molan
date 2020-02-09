def partition(smilist,ratio=0.7):
    """
    A function to create test/ train split list
    :param smilist: smiles (list)
    :param ratio: test set split fraction (float)
    Return type: traininglist, testlist (list)
    """
    from random import shuffle, random
    import numpy as np

    shuffle(smilist, random)
    trainlen = int(np.floor( len(smilist)*ratio ) ) 
    return smilist[0:trainlen],smilist[trainlen:]



def test_train_split_custers(dictionary,dbname='All',Ncluster=7,outlier=6):
    """
    A function to create clusterwise test/ train split json (i.e. Murtagh)
    
    Use the 'analyzecluster.py' to visualize the cluster and extract the outlier cluster

    :param dictionary : clustered json dictionary (dict)
    :param dbname: 'All' (default) or 'Enabradstrom'  (str)
    :param Ncluster:  (int)
    :param outlier: (int)
    Return type: clusterwise split traininglist, testlist (dict)
    """
    from test_train_split import partition
    subdict = dictionary['DB'][0][dbname][0]['clusters']
    for num in range(len(subdict) ):
        if subdict[num]['numcluster'] == Ncluster:
            testtrain_dict = dict()
            global_train = list()
            global_test = list()
            outlier_train = list()
            outlier_test = list()
            
            for clustnum in sorted(subdict[num]['val'].keys() ):
                clustdict= dict()
                clustdict = subdict[num]['val'][clustnum]
                testtrain_dict[int(clustnum)] = { 'train' : [], 'test' : []}
                local_smi = list()
                for smiindex in range(len(clustdict)):
                    local_smi.append(clustdict[smiindex]['smiles'] )
                    
                
                ltrain ,ltest  = partition(local_smi)
                [global_train.append(lsmi) for lsmi in ltrain]
                lsmi=''
                [global_test.append(lsmi) for lsmi in ltest]
                if int(clustnum) != outlier:
                    lsmi=''
                    [outlier_train.append(lsmi) for lsmi in ltrain]
                    lsmi=''
                    [outlier_test.append(lsmi) for lsmi in ltest]
                    
                
                [testtrain_dict[int(clustnum)]['train'].append(i) for i in ltrain]
                [testtrain_dict[int(clustnum)]['test'].append(j) for j in ltest]
            
            testtrain_dict['global'] = { 'train' : global_train, 'test' : global_test}
            #print(len(global_train),len(global_test))
            testtrain_dict['outlier'] = { 'train' : outlier_train, 'test' : outlier_test}
            #print(len(outlier_train),len(outlier_test))
                    
    return testtrain_dict
    


def test_train_split_butina(dictionary,dbname='All'):
    """
    A function to create clusterwise test/ train split json (i.e. Butina)
    :param dictionary : clustered json dictionary (dict)
    :param dbname: 'All' (default) or 'Enabradstrom'  (str)
    Return type: clusterwise split traininglist, testlist (dict)
    """
    from test_train_split import partition

    subdict = dictionary['DB'][0][dbname][0]['clusters']
    nclust = subdict[0]['numcluster']
    testtrain_dict = dict()
    global_train = list()
    global_test = list()

    for clustnum in range(1,nclust+1):
        clustdict= dict()
        clustdict = subdict[0]['val'][str(clustnum)]
        length = len(clustdict)
        if  length > 9 :
            local_smi = list()
            for smiindex in range(len(clustdict)):
                local_smi.append(clustdict[smiindex]['smiles'] )
            ltrain ,ltest  = partition(local_smi)
            [global_train.append(lsmi) for lsmi in ltrain]
            lsmi=''
            [global_test.append(lsmi) for lsmi in ltest]



    testtrain_dict['global'] = { 'train' : global_train, 'test' : global_test}
    return testtrain_dict
