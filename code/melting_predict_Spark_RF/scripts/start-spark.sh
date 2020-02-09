#! /bin/bash
set -u

[[ -z ${SPARKJOB_JOBID+X} ]] &&
	declare SPARKJOB_JOBID=$COBALT_JOBID    # Change it for other job system
export SPARKJOB_JOBID

if [[ -z ${SPARKJOB_HOST+X} ]];then
	declare -r host=$(hostname)
	if [[ $host =~ ^theta ]];then
		declare SPARKJOB_HOST=theta
	elif [[ $host =~ ^cooley ]];then
		declare SPARKJOB_HOST=cooley
	else
		echo "Cannot determine host type for this host: $host"
		exit 1
	fi
fi
export SPARKJOB_HOST

# Set the directory containing our scripts if unset.
# SPARKJOB_SCRIPTS_DIR is passed to the job via qsub.
[[ -z ${SPARKJOB_SCRIPTS_DIR+X} ]] &&
	declare SPARKJOB_SCRIPTS_DIR="$(cd $(dirname "$0")&&pwd)"
export SPARKJOB_SCRIPTS_DIR
[[ -z ${SPARKJOB_OUTPUT_DIR+X} ]] &&
	declare SPARKJOB_OUTPUT_DIR="$(pwd)"
export SPARKJOB_OUTPUT_DIR
[[ -z ${SPARKJOB_PYVERSION+X} ]] && declare -i SPARKJOB_PYVERSION=3
export SPARKJOB_PYVERSION
[[ -z ${SPARKJOB_INTERACTIVE+X} ]] && declare -i SPARKJOB_INTERACTIVE=0
export SPARKJOB_INTERACTIVE
[[ -z ${SPARKJOB_SCRIPTMODE+X} ]] && declare -i SPARKJOB_SCRIPTMODE=0
export SPARKJOB_SCRIPTMODE

source "$SPARKJOB_SCRIPTS_DIR/setup.sh"

[[ -d $SPARK_WORKER_DIR ]] || mkdir -p "$SPARK_WORKER_DIR"
[[ -d $SPARK_CONF_DIR ]] || mkdir -p "$SPARK_CONF_DIR"
[[ -d $SPARK_LOG_DIR ]] || mkdir -p "$SPARK_LOG_DIR"

case $SPARKJOB_HOST in
theta)
	aprun -n $COBALT_PARTSIZE -N 1 hostname | grep ^nid > "$SPARK_CONF_DIR/nodes"
	aprun -n 1 -N 1 \
		-e SPARKJOB_HOST="$SPARKJOB_HOST" \
		-e SPARKJOB_SCRIPTS_DIR="$SPARKJOB_SCRIPTS_DIR" \
		-e SPARKJOB_OUTPUT_DIR="$SPARKJOB_OUTPUT_DIR" \
		-e SPARKJOB_WORKING_DIR="$SPARKJOB_WORKING_DIR" \
		-e SPARKJOB_PYVERSION="$SPARKJOB_PYVERSION" \
		-e SPARKJOB_INTERACTIVE=$SPARKJOB_INTERACTIVE \
		-e SPARKJOB_SCRIPTMODE=$SPARKJOB_SCRIPTMODE \
		$SPARKJOB_SCRIPTS_DIR/run-spark.sh "$@"
	;;
cooley)
	cp "$COBALT_NODEFILE" "$SPARK_CONF_DIR/nodes"
	"$SPARKJOB_SCRIPTS_DIR/run-spark.sh" "$@" ;;
*)
	echo "Unknow host $SPARKJOB_HOST"; exit 1 ;;
esac
