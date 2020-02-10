[[ -d $SPARK_CONF_DIR ]] || mkdir -p "$SPARK_CONF_DIR"


# JVM support for Xeon TSX (coarse-grain locking) -XX:+UseRTMLocking
[[ -s $SPARK_CONF_DIR/spark-defaults.conf ]] ||
  cat > "$SPARK_CONF_DIR/spark-defaults.conf" <<'EOF'
spark.driver.memory              32g
spark.executor.memory            128g
#spark.driver.extraJavaOptions    -XX:+UseParallelGC -XX:ParallelGCThreads=16
#spark.executor.extraJavaOptions  -XX:+UseParallelGC -XX:ParallelGCThreads=16
EOF

unset PYSPARK_DRIVER_PYTHON
unset PYSPARK_DRIVER_PYTHON_OPTS

echo "env_local.sh"
