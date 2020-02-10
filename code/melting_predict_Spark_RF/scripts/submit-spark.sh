#! /bin/bash
set -u

# Set the directory containing our scripts if unset.
[[ -z ${SPARKJOB_SCRIPTS_DIR+X} ]] && \
	declare SPARKJOB_SCRIPTS_DIR="$(cd $(dirname "$0")&&pwd)"

declare -r version='SPARK JOB  v1.0.0'
declare -r usage="$version"'

Usage:
	submit-spark.sh [options] [JOBFILE [arguments ...]]

JOBFILE (optional) can be:
	script.py		pyspark scripts
	bin.jar			java binaries
	run-example CLASS	run spark example CLASS
	scripts			other executable scripts

Required options:
	-A PROJECT		Allocation name
	-t WALLTIME		Max run time
	-n NODES		Job node count
	-q QUEUE		Queue name

Optional options:
	-o OUTPUTDIR		Directory for COBALT output files (default: current dir)
	-w WAITTIME		Time to wait for prompt in minutes (default: 30)
	-s			Enable script mode
	-I			Start an interactive ssh session
	-m			Master uses a separate node
	-p <2|3>		Python version (default: 3)
	-h			Print this help message

Example:
	./submit-spark.sh -A datascience -t 60 -n 2 -q pubnet-debug -w 10
	./submit-spark.sh -A datascience -t 60 -n 2 -q debug $SPARK_HOME/examples/src/main/python/pi.py 10000
	./submit-spark.sh -A datascience -t 60 -n 2 -q debug -s SomeExe Args
'

while getopts hmsIA:t:n:q:w:p:o: OPT; do
	case $OPT in
	I)	declare -ir	interactive=1;;
	s)	declare -ir	scriptMode=1;;
	A)	declare -r	allocation="$OPTARG";;
	t)	declare -r	time="$OPTARG";;
	n)	declare -r	nodes="$OPTARG";;
	q)	declare -r	queue="$OPTARG";;
	w)	declare -ir	waittime=$((OPTARG*60));;
	p)	declare -ir	pyversion=$OPTARG;;
	o)	declare -r	outputdir="$OPTARG";;
	m)	declare -ir	separate_master=1;;
	h)	echo "$usage"; exit 0;;
	?)	echo "$usage"; exit 1;;
	esac
done

[[ -z ${waittime+X} ]] && declare -ir waittime=$((30*60))
[[ -z ${pyversion+X} ]] && declare -ir pyversion=3
[[ -z ${scriptMode+X} ]] && declare -ir scriptMode=0
[[ -z ${outputdir+X} ]] && declare -r outputdir=.
[[ -z ${separate_master+X} ]] && declare -ir separate_master=0

if [[ -z ${allocation+X} || -z ${time+X} || -z ${nodes+X} || -z ${queue+X} ]];then
	echo "$usage"
	exit 1
fi

if ((pyversion != 2 && pyversion != 3));then
	echo "Preconfigured Python version can only be 2 or 3,"
	echo "but got $pyversion."
	echo "Using your custom python version."
	echo "Make sure to set it up for compute nodes."
fi

shift $((OPTIND-1))

declare -a scripts=()

if (($#>0));then
	if [[ -s $1 || $1 == run-example ]];then
		[[ -z ${interactive+X} ]] && declare -ir interactive=0
		scripts=( "$@" )
		echo "# Submitting job: ${scripts[@]}"
	else
		echo "File does not exist: $1"
		exit 1
	fi
else
	[[ -z ${interactive+X} ]] && declare -ir interactive=1
	echo "Submitting an interactive job and wait for at most $waittime sec."
fi

declare -r host=$(hostname)
if [[ $host =~ ^theta ]];then
	declare SPARKJOB_HOST=theta
elif [[ $host =~ ^cooley ]];then
	declare SPARKJOB_HOST=cooley
else
	echo "Cannot determine host type for this host: $host"
	exit 1
fi

if [[ ! -d $outputdir ]];then
	if ! mkdir "$outputdir";then
		echo "Cannot create directory: $outputdir"
		exit 1
	fi
fi

cd "$outputdir"
declare SPARKJOB_OUTPUT_DIR="$(pwd)"
declare SPARKJOB_PYVERSION=$pyversion
declare SPARKJOB_INTERACTIVE=$interactive
declare SPARKJOB_SCRIPTMODE=$scriptMode
declare SPARKJOB_SEPARATE_MASTER=$separate_master

declare -i SPARKJOB_JOBID=0
mysubmit() {
	# Options to pass to qsub
	local -a opt=(
		-n $nodes -t $time -A $allocation -q $queue
		--env "SPARKJOB_HOST=$SPARKJOB_HOST"
		--env "SPARKJOB_SCRIPTS_DIR=$SPARKJOB_SCRIPTS_DIR"
		--env "SPARKJOB_PYVERSION=$SPARKJOB_PYVERSION"
		--env "SPARKJOB_INTERACTIVE=$SPARKJOB_INTERACTIVE"
		--env "SPARKJOB_SCRIPTMODE=$SPARKJOB_SCRIPTMODE"
		--env "SPARKJOB_OUTPUT_DIR=$SPARKJOB_OUTPUT_DIR"
		--env "SPARKJOB_SEPARATE_MASTER=$SPARKJOB_SEPARATE_MASTER"
		-O "$SPARKJOB_OUTPUT_DIR/\$jobid"
		"$SPARKJOB_SCRIPTS_DIR/start-spark.sh"
	)
	case $SPARKJOB_HOST in
	theta)	opt=(--attrs 'enable_ssh=1' "${opt[@]}") ;;
	esac
	if ((${#scripts[@]}>0));then
		opt+=("${scripts[@]}")
	fi
	SPARKJOB_JOBID=$(qsub "${opt[@]}")
	if ((SPARKJOB_JOBID > 0));then
		echo "# Submitted"
		echo "SPARKJOB_JOBID=$SPARKJOB_JOBID"
	else
		echo "# Submitting failed."
		exit 1
	fi
}

if ((interactive>0));then
	cleanup(){ ((SPARKJOB_JOBID>0)) && qdel $SPARKJOB_JOBID; } 
	trap cleanup 0
	mysubmit
	declare -i mywait=10 count=0
	echo "Waiting for Spark to launch..."
	source "$SPARKJOB_SCRIPTS_DIR/setup.sh"
	for ((count=0;count<waittime;count+=mywait));do
		[[ ! -s $SPARKJOB_WORKING_ENVS ]] || break
		sleep $mywait
	done
	if [[ -s $SPARKJOB_WORKING_ENVS ]];then
		source "$SPARKJOB_SCRIPTS_DIR/setup.sh"	# pull in spark envs
		echo "# Spark is now running (SPARKJOB_JOBID=$SPARKJOB_JOBID) on:"
		column "$SPARK_CONF_DIR/slaves" | sed 's/^/# /'
		declare -p SPARK_MASTER_URI
		declare -ar sshmaster=(ssh -o ControlMaster=no -t $MASTER_HOST)
		declare -r runbash="exec bash --rcfile <(
			echo SPARKJOB_JOBID=\'$SPARKJOB_JOBID\';
			echo SPARKJOB_HOST=\'$SPARKJOB_HOST\';
			echo SPARKJOB_SCRIPTS_DIR=\'$SPARKJOB_SCRIPTS_DIR\';
			echo SPARKJOB_PYVERSION=\'$SPARKJOB_PYVERSION\';
			echo SPARKJOB_INTERACTIVE=\'$SPARKJOB_INTERACTIVE\';
			echo SPARKJOB_SCRIPTMODE=\'$SPARKJOB_SCRIPTMODE\';
			echo SPARKJOB_OUTPUT_DIR=\'$SPARKJOB_OUTPUT_DIR\';
			echo SPARKJOB_SEPARATE_MASTER=\'$SPARKJOB_SEPARATE_MASTER\';
			echo source ~/.bashrc;
			echo source \'$SPARKJOB_SCRIPTS_DIR/setup.sh\')
			-l -i"
		echo "# Spawning bash on host: $MASTER_HOST"
		case $SPARKJOB_HOST in
		theta)
			ssh -o ControlMaster=no -t thetamom$((1+RANDOM%3)) \
				"${sshmaster[@]}" "\"$runbash\""
			;;
		cooley)
			"${sshmaster[@]}" "$runbash"
			;;
		*)
			echo "Unknown host: $SPARKJOB_HOST"
			exit 1
		esac
	else
		echo "Spark failed to launch within $((waittime/60)) minutes."
	fi
else
	mysubmit
fi
