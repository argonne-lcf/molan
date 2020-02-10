def plot_cluster(dictionary, numcluster):
    """
    A function to analyze/plot the observable assigned to the unsupervised learned chemistry aware  clusters
    :param dictionary: the clustered json (dict)
    :param numcluster: the number of cluster to plot (int)
    Return type: matplotlib figure   
    """
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
 
    fig = plt.figure(figsize=(20, 24))
    
    for  key in ['Enabradstrom','All']:
        print("Probing the DB: {}".format(key))
        subdict = dictionary['DB'][0][key][0]['clusters']
        for num in range(len(subdict) ):
            if key == 'Enabradstrom':
                clustersize  = (subdict[num]['numcluster'])
                if clustersize == numcluster:
                    plotdict = subdict[num]['val']
                    c=1
                    for pnum in range(1,numcluster+1): 
                        mp_list = list()
                        for values in plotdict[str(c)] : #c goes from 1:ncluster 
                            mp_list.append(values['MP'])
                        mp_array = np.array(mp_list).astype(np.float)
                        ax = fig.add_subplot(2, numcluster, pnum)
                        ax = sns.distplot(mp_array, bins=20, kde=False, fit=stats.norm,color='g');
                        plt.rc('legend', fontsize=16)
                        #plt.rc('ylabel', fontsize=12)
                        # Get the fitted parameters used by sns
                        (mu, sigma) = stats.norm.fit(mp_array)
                        print "cluster={0}, mu={1}, sigma={2}, samplesize={3}".format(c,mu, sigma,len(mp_array))
                        # Legend and labels 
                        if pnum == 1:
                            plt.legend(["normal dist. fit ($\mu=${0:.0f}, $\sigma=${1:.2f})".format(mu, sigma)])
                            plt.ylabel('Frequency')
                        else:
                            plt.legend([" ($\mu=${0:.0f}, $\sigma=${1:.2f})".format(mu, sigma)])
                            
                        # Cross-check this is indeed the case - should be overlaid over black curve
                        #x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
                        #ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
                        #plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),
                        #"cross-check"])

                        c+= 1    
            else:
                clustersize  = (subdict[num]['numcluster'])
                if clustersize == numcluster:
                    plotdict = subdict[num]['val']
                    c=1
                    for pnum in range(numcluster+1,2*numcluster+1):
                        mp_list = list()
                        for values in plotdict[str(c)]:
                            mp_list.append(values['MP'])
                        mp_array = np.array(mp_list).astype(np.float)
                        ax = fig.add_subplot(2, numcluster, pnum)
                        ax = sns.distplot(mp_array, bins=20, kde=False, fit=stats.norm,color='b');
                        # Get the fitted parameters used by sns
                        (mu, sigma) = stats.norm.fit(mp_array)
                        print "cluster={0},mu={1}, sigma={2},samplesize={3}".format(c,mu, sigma,len(mp_array))
                        # Legend and labels 
                        if pnum == numcluster+1:
                            plt.legend(["normal dist. fit ($\mu=${0:.0f}, $\sigma=${1:.2f})".format(mu, sigma)])
                            plt.ylabel('Frequency')
                        else:
                            plt.legend(["($\mu=${0:.0f}, $\sigma=${1:.2f})".format(mu, sigma)])
                        # Cross-check this is indeed the case - should be overlaid over black curve
                        #x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
                        #ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
                        #plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma),
                        #"cross-check"])
                        
                        c+=1    
    

    

    plt.show()
    return plt

