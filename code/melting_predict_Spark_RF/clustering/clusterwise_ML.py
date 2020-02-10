#!/data/ganesh/Software/anaconda/bin/python -tt
# E-MAIL:-  gsivaraman@anl.gov
# Created on Sep 17, 2018 9:20 PM
# import modules used here
from pyspark import SparkContext, SparkConf
import os
import json
import numpy as np
import argparse
from test_train_split import test_train_split_custers
from test_train_split import test_train_split_butina
from get_metadata import gen_numpy
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import SQLContext
from time import time


def get_argparse():
    """
        A function to parse all the input arguments
    """
    parser = argparse.ArgumentParser(description='A python script to train/validate Chemistry Aware ML models on 47K melting temperature dataset')
    parser.add_argument('-p','--path', type=str,metavar='',required=True, help="Full path to the source folder of the 47K JSON's")
    parser.add_argument('-cj','--clusterjson', type=str,metavar='',required=True, help="Point to the source file of clusters of the 47K JSON's")
    parser.add_argument('--cores', help="cores per node", type=int,metavar='', default=12)
    parser.add_argument('--nodes', help="nodes on cluster", type=int, metavar='',default=1)
    parser.add_argument('-np','--npart', help="Increase this parameter incase array serialization task is too large", type=int, metavar='',default=1)
    parser.add_argument('-cv','--crossval',help='Perform hyper paramater tuning: Yes or No',type=str,metavar='',required=True,default='No')
    return parser.parse_args()


def labelDF(train_XY, test_XY,args,sc, sqlContext):
    """
        A function to Construct LabeledPoint dataframes for Spark ML supervised learning
        Spits out  70:30 split dataframes for training and testing
        :param train_XY, test_XY: Numpy array
        :param args: args
        :param sc: Spark context
        :param  sqlContext: Spark SQL context
        :return: train_df,test_df (dataframe)
    """
    
    train_rdd = sc.parallelize(train_XY, args.cores * args.nodes * args.npart)
    test_rdd = sc.parallelize(test_XY, args.cores * args.nodes * args.npart)
    print('Training point generation begins now')
    training_points = train_rdd.map(lambda row: ( float(row[-1]), Vectors.dense( row[:-1]) ) )
    test_points = test_rdd.map(lambda row: ( float(row[-1]), Vectors.dense( row[:-1]) ) )
    train_df = sqlContext.createDataFrame(training_points, ["label", "features"])
    test_df = sqlContext.createDataFrame(test_points, ["label", "features"])
    # Split the data into training and test sets (30% held out for testing)
    #(trainingData, testData) = training_df.randomSplit([0.7, 0.3])
    return train_df,test_df


def bootstrap():
    """
        Spark configuration
        :return: sc, sqlContext
    """
    sconf = SparkConf()
    sc = SparkContext(conf=sconf)
    sqlContext = SQLContext(sc)
    return sc,  sqlContext


def RF(trainingData, testData):
    """
        Random Forest Tree Regression Model
        :param trainingData:
        :param testData:
        :param args
        :return: Trained model, predictions, nt (int), md (int)
        """
    nt,md=120,20
    rf = RandomForestRegressor( numTrees=nt, featureSubsetStrategy="auto",\
                               impurity='variance', maxDepth=md, maxBins=100) #120,20
    model = rf.fit(trainingData)
    predictions = model.transform(testData)
    return model, predictions, nt, md


def RF_CV(trainingData, testData):
    """
    Random Forest Tree Regression 5-fold CV Model 
    :param trainingData:
    :param testData:
    :return: Best Trained model, predictions
    """
    rf = RandomForestRegressor( featureSubsetStrategy="auto",\
                                    impurity='variance', maxBins=100)
    paramGrid = ParamGridBuilder()\
        .addGrid(rf.numTrees, [10, 20, 30, 40, 50, 100,120, 140 ]) \
        .addGrid(rf.maxDepth, [2, 6, 8, 10, 12, 14, 16, 18, 20])\
        .build()

    cvs = CrossValidator(estimator=rf,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               numFolds=5)

    model = cvs.fit(trainingData)
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

    dictionary = json.load(open(args.clusterjson) )
    print "\nReading data for algorithm: ", dictionary["algorithm"]
    if dictionary["algorithm"] == "Butina":
        testtrain_dict = test_train_split_butina(dictionary,dbname='All')
    elif dictionary["algorithm"] == "Murtagh" :
        testtrain_dict = test_train_split_custers(dictionary,dbname='All',Ncluster=5,outlier=2)
    with open('testtrain_dict.json', 'w') as f:
        json.dump(testtrain_dict, f,indent=4)
    print "\nTraining set size is: ",len(testtrain_dict['global']['train'])
    print "\nTest set size is : ", len(testtrain_dict['global']['test'])
    output = {}
    for key in testtrain_dict.keys():
            print "\nStarting with processing cluster :", key
            output[key] = {'metrics' :[], 'labelsandprediction' : []}
            train_xy = gen_numpy(args.path,testtrain_dict[key]['train'])
            test_xy = gen_numpy(args.path,testtrain_dict[key]['test'])
            trainingData, testData = labelDF(train_xy, test_xy,args,sc, sqlContext)
             if args.crossval == 'No':
                model, predictions, nt, md = RF(trainingData, testData)
            elif args.crossval == 'Yes':
                model, predictions = RF_CV(trainingData, testData)
            predictson = predictions.select("prediction","label").toJSON().collect()
            rmse, mae, r2_score = get_metrics(predictions)
            mlist = [{'RMSE': rmse}, {'MAE': mae},{'R2': r2_score}, {'Unit': 'K'}, {'property': 'MP'}  ]
            print "\n", mlist
            output[key]= {'metrics' : mlist, 'labelsandprediction' : predictson}
            print "\nCompleted the processing of  cluster :",key
    
    
    
    # Writing JSON data
    with open('clusterwise_results.json', 'w') as f:
        json.dump(output, f,indent=4)


