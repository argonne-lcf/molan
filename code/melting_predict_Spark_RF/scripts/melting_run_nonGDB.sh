#! /bin/bash

./submit-spark.sh -A datascience -t 480  -n 8  -q  pubnet   Predict_Melting.py  -p  /path/to/Melting/dataset/   -db  All    --nodes 8  --cores 12 --npart 4  -m  RF  -d  Morgan2DCMSE    -b No  -cv  No 
