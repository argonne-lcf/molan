# SPARK
export SPARK_HOME=/soft/datascience/apache_spark

export SPARK_WORKER_DIR="$SPARKJOB_WORKING_DIR/workers"
export SPARK_CONF_DIR="$SPARKJOB_WORKING_DIR/conf"
export SPARK_LOG_DIR="$SPARKJOB_WORKING_DIR/logs"

# Java
module load java

# Python
if ((SPARKJOB_PYVERSION==2));then
	module load intelpython27
elif ((SPARKJOB_PYVERSION==3));then
	module load intelpython35
fi	# else you are on your own.

export PYSPARK_PYTHON="$(which python)"

if ((SPARKJOB_INTERACTIVE>0));then
	echo "Remember to set PYSPARK_DRIVER_PYTHON and PYSPARK_DRIVER_PYTHON_OPTS,"
	echo "if you want to run jupyter."
fi
