# E-MAIL:-  gsivaraman@anl.gov
# import modules used here
import argparse
from pyspark import SparkContext, SparkConf
import os
import json
import numpy as np
import CoulombMatrix as CM
from rdkit import Chem
from rdkit.Chem import AllChem
import pybel
import MoleculeGDB
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SQLContext
from time import time



def get_argparse():
    """
    A function to parse all the input arguments
    """
    parser = argparse.ArgumentParser(description='A python script to train/validate ML models on 47K melting temperature dataset. Includes a benchmark datasets as well:  GDB9.')
    parser.add_argument('-db','--database', type=str, metavar='',required=True,\
                        help='Select one of the following databases to compute: All, OCHEM, Enamine, Bradstrom, Enabradstrom, GDB9', default='Bradstrom')
    parser.add_argument('-p','--path', type=str,metavar='',required=True, help="Full path to the source folder of the 47K JSON's")
    parser.add_argument('-m','--MLalg', help="Training algorithms: RF, GBT", type=str, metavar='', default='RF')
    parser.add_argument('-c','--cores', help="cores per node", type=int,metavar='', default=12)
    parser.add_argument('-n','--nodes', help="nodes on cluster", type=int, metavar='',default=1)
    parser.add_argument('-np','--npart', help="Increase this parameter incase array serialization task is too large", type=int, metavar='',default=1)
    parser.add_argument('-d','--descriptor', help="Chemical descritor: CM, CMSE, Morgan2D, Morgan2DSE, Morgan2DSEext, Morgan2DCMSE ", type=str, metavar='',default='Morgan2D')
    parser.add_argument('-cv','--crossval',help='Perform hyper paramater tuning: Yes or No',type=str,metavar='',required=True,default='No')
    parser.add_argument('-b','--benchmark',help='Perform benchmark: Yes or No',type=str,metavar='',required=True,default='No')
#    parser.add_argument('-b','--bset', type=str,help='GDB8, GDB9', default='GDB8')



    #group = parser.add_mutually_exclusive_group()
    #group.add_argument('-q','--quiet',action='store_true', help='print quiet')
    #group.add_argument('-v','--verbose',action='store_true', help='print verbose')
    return parser.parse_args()

#args= parser.parse_args()


def get_metadata(args):
    """
    A function to extract the dataset size related  parameters and the numpy array required for ML.
    :param args: args
    :return: train_XY (Numpy array), trainlen (int), maxdim (int)
    """
    global jlist
    json_list = []
    bradstrom_list = []
    OCHEM_list = []
    Enamine_list = []
    for filename in os.listdir(args.path):
        if 'json' in filename:
            json_list.append(filename)
            if ('Bradley' in filename or 'Bergstrom' in filename):
                bradstrom_list.append(filename)
            elif 'OCHEM' in filename:
                OCHEM_list.append(filename)
            elif 'Enamine' in filename:
                Enamine_list.append(filename)

    if args.database == 'All':
        jlist = json_list[:]
    elif args.database == 'Bradstrom':
        jlist = bradstrom_list[:]
    elif args.database == 'OCHEM' :
        jlist = OCHEM_list[:]
    elif args.database == 'Enamine':
        jlist = Enamine_list[:]
    elif args.database == 'Enabradstrom':
        jlist = Enamine_list + bradstrom_list

    index_list = []
    for item in jlist:
        sample = json.load(open(args.path+item))
        index = len(sample['gaussian']['coords'])
        index_list.append(index)

    trainlen = len(index_list)
    maxdim = max(index_list)

    if args.descriptor == 'CM':
        train_XY = np.zeros((trainlen, maxdim * maxdim + 1))
        local_X = np.zeros((maxdim * maxdim + 1))

        for num, jsonfile in enumerate(jlist):
            xyzheader, chargearray, xyzij = CM.process_json(args.path + jsonfile)
            local_X[:-1] = CM.CoulombMatDescriptor(maxdim, xyzheader, chargearray, xyzij)
            currentjson = json.load(open(args.path + jsonfile))
            local_X[-1] = currentjson['Tm']
            train_XY[num, :] = local_X


    elif args.descriptor == 'CMSE':
        train_XY = np.zeros((trainlen, maxdim * maxdim + 2))
        local_X = np.zeros((maxdim * maxdim + 2))

        for num, jsonfile in enumerate(jlist):
            xyzheader, chargearray, xyzij = CM.process_json(args.path + jsonfile)
            local_X[:-2] = CM.CoulombMatDescriptor(maxdim, xyzheader, chargearray, xyzij)
            currentjson = json.load(open(args.path + jsonfile))
            local_X[-2] = currentjson['gaussian']['properties']['SMD_solvation_energy']
            local_X[-1] = currentjson['Tm']
            train_XY[num, :] = local_X

    elif args.descriptor == 'Morgan2D':
        nb = 1024
        train_XY = np.zeros((trainlen, nb + 1))
        local_X = np.zeros((nb + 1))

        for num, jsonfile in enumerate(jlist):
            currentjson = json.load(open(args.path + jsonfile))
            mol = Chem.MolFromSmiles(currentjson['smiles'])
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nb)  # type:
            local_X[0:nb] = ecfp
            local_X[-1] = currentjson['Tm']
            train_XY[num, :] = local_X

    elif args.descriptor == 'Morgan2DSE':
        nb = 1024
        train_XY = np.zeros((trainlen, nb + 2))
        local_X = np.zeros((nb + 2))

        for num, jsonfile in enumerate(jlist):
            currentjson = json.load(open(args.path + jsonfile))
            mol = Chem.MolFromSmiles(currentjson['smiles'])
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nb)  # type:
            local_X[0:nb] = ecfp
            local_X[-2] = currentjson['gaussian']['properties']['SMD_solvation_energy']
            local_X[-1] = currentjson['Tm']
            train_XY[num, :] = local_X


    elif args.descriptor == 'Morgan2DSEext':
        nb = 1024
        train_XY = np.zeros((trainlen, nb + 5))
        local_X = np.zeros((nb + 5))

        for num, jsonfile in enumerate(jlist):
            currentjson = json.load(open(args.path + jsonfile))
            mol = Chem.MolFromSmiles(currentjson['smiles'])
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nb)  # type:
            local_X[0:nb] = ecfp
	    local_X[-5] = currentjson['gaussian']['properties']['total_energy'] 
	    local_X[-4] = currentjson['gaussian']['properties']['electric_dipole_moment_norm']
	    local_X[-3] = currentjson['gaussian']['properties']['electric_quadrupole_moment_norm']
            local_X[-2] = currentjson['gaussian']['properties']['SMD_solvation_energy']
            local_X[-1] = currentjson['Tm']
            train_XY[num, :] = local_X


    elif args.descriptor == 'Morgan2DCMSE':
        nb = 1024
        train_XY = np.zeros((trainlen, maxdim * maxdim + nb + 2))
        local_X = np.zeros((maxdim * maxdim + nb + 2))

        for num, jsonfile in enumerate(jlist):
            currentjson = json.load(open(args.path + jsonfile))
            mol = Chem.MolFromSmiles(currentjson['smiles'])
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nb)
            local_X[0:nb] = ecfp
            xyzheader, chargearray, xyzij = CM.process_json(args.path + jsonfile)
            local_X[nb:-2] = CM.CoulombMatDescriptor(maxdim, xyzheader, chargearray, xyzij)
            local_X[-2] = currentjson['gaussian']['properties']['SMD_solvation_energy']
            local_X[-1] = currentjson['Tm']
            train_XY[num, :] = local_X


    return train_XY, trainlen, maxdim


def get_gdb(args):
    """
    A function to extract the dataset size related  parameters and the numpy array required for ML benchmark.
    :param args: args
    :return: train_XY (Numpy array), trainlen (int), maxdim (int)
    """
    ##--> Import GDB9
    xyz_file = [args.path + f for f in os.listdir(args.path)]
    natom_array = []
    for e in sorted(xyz_file):
        mols = MoleculeGDB.MoleculeGDB9()
        fname = e
        mols.readxyz(fname)
        natom_array.append(len(mols.coords['coords']))
    maxdim = max(natom_array)
    trainlen = len(xyz_file)

    if args.descriptor == 'CM':
    	train_XY = np.zeros((trainlen, maxdim * maxdim + 1))
    	local_X = np.zeros((maxdim * maxdim +1))
    	count = 0
    	for num, e in enumerate(sorted(xyz_file)):
            mols = MoleculeGDB.MoleculeGDB9()
            fname = e
            mols.readxyz(fname)
            xyzheader, chargearray, xyzij = CM.process_gdb_json(mols)
            local_X[:-1] = CM.CoulombMatDescriptor(maxdim, xyzheader, chargearray, xyzij)
            local_X[-1] = float(mols.json['homo'])
            train_XY[num, :] = local_X

    elif args.descriptor == 'Morgan2D':
        nb = 1024
        train_XY = np.zeros((trainlen, nb + 1))
        local_X = np.zeros((nb + 1))

        for num, e in enumerate(sorted(xyz_file)):
            mols = MoleculeGDB.MoleculeGDB9()
            fname = e
            mols.readxyz(fname)
            cansmile = pybel.readstring("smi", mols.smiles[1]).write("can").strip("\t\n")
            mol = Chem.MolFromSmiles(cansmile)
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nb)  # type:
            local_X[0:nb] = ecfp
            local_X[-1] = float(mols.json['gap'])
            train_XY[num, :] = local_X

    return train_XY, trainlen, maxdim



def labelDF(train_XY,args,sc, sqlContext):
    """
    A function to Construct LabeledPoint dataframes for Spark ML supervised learning
    Spits out  70:30 split dataframes for training and testing
    :param train_XY: Numpy array
    :param args: args
    :param sc: Spark context
    :param  sqlContext: Spark SQL context
    :return: trainingData, testData
    """

    rdd = sc.parallelize(train_XY, args.cores * args.nodes * args.npart)
    print('Training point generation begins now')
    training_points = rdd.map(lambda row: ( float(row[-1]), Vectors.dense( row[:-1]) ) ) #labelData(rdd)
    training_df = sqlContext.createDataFrame(training_points, ["label", "features"])
    # Split the data into training and test sets (30% held out for testing)
    (trainingData, testData) = training_df.randomSplit([0.7, 0.3])
    return trainingData, testData

def bootstrap():
    """
    Spark configuration
    :return: sc, sqlContext
    """
    sconf = SparkConf()
    sc = SparkContext(conf=sconf)
    sqlContext = SQLContext(sc)
    return sc,  sqlContext


def GBT_CV(trainingData, testData):
    """
    Gradient Boosted Tree Regression Model Selection
    :param trainingData:
    :param testData:
    :return: Trained model, predictions
    """
    gbt = GBTRegressor(seed=42)
    paramGrid = ParamGridBuilder()\
        .addGrid(gbt.maxIter, [50, 100, 200, 300, 400, 500 ]) \
        .addGrid(gbt.maxDepth, [2, 6, 10, 14])\
        .build()

    tvs = TrainValidationSplit(estimator=gbt,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)
    model = tvs.fit(trainingData)
    predictions = model.transform(testData)
    return model, predictions


def GBT(trainingData, testData):
    """
     Gradient Boosted Tree Regression Model
    :param trainingData:
    :param testData:
    :return: Trained model, predictions
    """
    gbt = GBTRegressor( maxIter=100, maxDepth=6, seed=42)
    model = gbt.fit(trainingData)
    predictions = model.transform(testData)
    return model, predictions


def RF(trainingData, testData, args):
    """
    Random Forest Tree Regression Model
    :param trainingData:
    :param testData:
    :param args
    :return: Trained model, predictions, nt (int), md (int)
    """
    if (args.descriptor == 'CM' or args.descriptor == 'CMSE' or args.descriptor == 'Morgan2DCMSE'):
	nt,md=50,14  
    elif (args.descriptor == 'Morgan2D' or args.descriptor == 'Morgan2DSE' or args.descriptor == 'Morgan2DSEext'):
	nt,md=120,20
    rf = RandomForestRegressor( numTrees=nt, featureSubsetStrategy="auto",\
                                    impurity='variance', maxDepth=md, maxBins=100) #120,20
    model = rf.fit(trainingData)
    predictions = model.transform(testData)
    return model, predictions, nt, md


def RF_CV(trainingData, testData):
    """
    Random Forest Tree Regression Model Selection
    :param trainingData:
    :param testData:
    :return: Trained model, predictions
    """
    rf = RandomForestRegressor( featureSubsetStrategy="auto",\
                                    impurity='variance', maxBins=100)
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.numTrees, [10, 20, 30, 40, 50, 100 ]) \
        .addGrid(rf.maxDepth, [2, 6, 8, 10, 12, 14])\
        .build()

    tvs = TrainValidationSplit(estimator=rf,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)
    model = tvs.fit(trainingData)
    predictions = model.transform(testData)
    return model, predictions


def get_metrics(predictions):
    rmse_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    rmse = rmse_evaluator.evaluate(predictions)

    mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    mae = mae_evaluator.evaluate(predictions)
    r2_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
    r2_score = r2_evaluator.evaluate(predictions)
    return rmse,mae, r2_score


if __name__=='__main__':
    t0 = time()
    args = get_argparse()
    sc,  sqlContext = bootstrap()
    if args.benchmark == 'No':
        train_XY, trainlen, maxdim = get_metadata(args)
    elif args.benchmark == 'Yes':
        if args.database == 'GDB9' :
            train_XY, trainlen, maxdim = get_gdb(args)
        else:
            print("Please chose  GDB9!")

    print("The  size of  the "+args.database+" dataset is {}, with a maximum of {} atoms".format(trainlen,maxdim) )
    trainingData, testData = labelDF(train_XY,args,sc, sqlContext)
    init_time = time() - t0
    print("######################################################################\n")
    print("\nRunning "+ args.MLalg +" Regressor with "+ args.descriptor +" descriptor for " + args.database + " dataset\n")
    print("######################################################################\n")
    print "Initialization and arrays generated in {} seconds".format(round(init_time, 3))


    t0 = time()
    if args.MLalg == 'GBT':
        if args.crossval == 'No':
            model, predictions = GBT(trainingData, testData)
        elif args.crossval == 'Yes':
            model, predictions = GBT_CV(trainingData, testData)
    elif args.MLalg == 'RF':
        if args.crossval == 'No':
            model, predictions, nt, md = RF(trainingData, testData,args)
        elif args.crossval == 'Yes':
            model, predictions = RF_CV(trainingData, testData)

    training_time = time() - t0
    print args.MLalg + " Model training and prediction metrics completed in {} seconds".format(round(training_time, 3))
    print('\n')


    t0 = time()
    rmse, mae, r2_score = get_metrics(predictions)
    pred_time = time() - t0

    print args.MLalg + " Prediction metrics completed in {} seconds".format(round(pred_time, 3))

    print("\nRoot Mean Squared Error (RMSE) on test data = %g K" % rmse)
    print("\nMean Absolute Error (MAE) on test data = %g K" % mae)
    print("\nR^2 = %g K" % r2_score)


    #Write a summary of information to JSON
    output = {}
    output['Max_Natom'] = maxdim
    output['trainingset_size'] = trainlen
    output['MLalg'] = args.MLalg
    if args.MLalg == 'RF':
    	output['RF_params'] = [{'numTrees': nt}, {'maxDepth': md}]
    output['Descriptor'] = args.descriptor
    output['Runtime'] = [{'Initalization': init_time}, {'Training': training_time}, {'Predictions': pred_time}, {'unit': 'sec'}]
    if args.database != 'GDB9':
    	output['metrics'] = [{'RMSE': rmse}, {'MAE': mae},{'R2': r2_score}, {'Unit': 'K'}, {'property': 'MP'}  ]
    else:
	output['metrics'] = [{'RMSE': rmse}, {'MAE': mae},{'R2': r2_score}, {'Unit': 'Ha'}, {'property': 'HOMO'} ]
    output['system'] = args.MLalg + "_" + args.descriptor + "_" + args.database
    output['dataset'] = args.database
    output['nodes'] = args.nodes
    # Writing JSON data
    with open(args.MLalg + "_" + args.descriptor + "_" + args.database+ '.json', 'w') as f:
        json.dump(output, f,indent=4)

    model.save(args.MLalg + "_" + args.descriptor + "_" + args.database + "_db")

    print("######################################################################\n")
    print("\nCompleted " + args.MLalg +" Regressor for " + args.database + " dataset\n")
    print("######################################################################\n")




