# SPARK
export SPARK_HOME=/soft/datascience/apache_spark

export SPARK_WORKER_DIR="$SPARKJOB_WORKING_DIR/workers"
export SPARK_CONF_DIR="$SPARKJOB_WORKING_DIR/conf"
export SPARK_LOG_DIR="$SPARKJOB_WORKING_DIR/logs"

# Java
export JAVA_HOME=/soft/compilers/java/jdk1.8.0_60
export PATH="$JAVA_HOME/bin:$PATH"

# Python
#if ((SPARKJOB_PYVERSION==2));then
#	export ANACONDA=/soft/libraries/anaconda
#elif ((SPARKJOB_PYVERSION==3));then
#	export ANACONDA=/soft/interpreters/python/anaconda/anaconda3/4.0.0
#fi	# else you are on your own.
export ANACONDA=/home/ganesh/miniconda2
export PYTHONPATH="$ANACONDA/bin/python"
export PATH="$ANACONDA/bin:$PATH"
export PYSPARK_PYTHON=$PYTHONPATH

if ((SPARKJOB_INTERACTIVE>0));then
	export PYSPARK_DRIVER_PYTHON=jupyter
	export PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --ip=$(hostname).cooley.pub.alcf.anl.gov --port=8002"
fi
