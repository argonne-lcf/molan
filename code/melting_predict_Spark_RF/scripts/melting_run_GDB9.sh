#! /bin/bash

./submit-spark.sh -A datascience -t 240  -n 4  -q  pubnet   Predict_Melting.py  -p /path/to/GDB9/   -db  GDB9    --nodes 4  --cores 12 --npart 4  -m  RF  -d CM   -b Yes  -cv  No 
