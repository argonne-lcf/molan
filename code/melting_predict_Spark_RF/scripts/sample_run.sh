#! /bin/bash
./submit-spark.sh -A datascience -t 30 -n 2 -q pubnet ../simplemap-spark-python/spark_benchmark_hpc.py  --generate --blocks 120 --block_size 10 --cores 12 --nodes 2 --json demo-for-ganesh.json
