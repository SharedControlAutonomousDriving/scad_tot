#!/bin/zsh

# Performs additional setup for the venv.
#
# Usage:
#    ./scripts/setup_venv.sh
#    ./scripts/setup_venv.sh -m CUSTOM_MARABOU_PATH
#
# Options: 
#   - m: custom path to marabou (default is .marabou)

PROJECT_PATH="$(pwd)"
PATH_TO_MARABOU="$(pwd)/.marabou"
_VENV_ACTIVATE="./venv/bin/activate"
_VENV_ACTIVATE_BAK="$_VENV_ACTIVATE.bak"
_VENV_ACTIVATE_DUMMY="$_VENV_ACTIVATE.dummy"
_VENV_ACTIVATE_TMP="./venv/bin/activate.tmp"
_VENV_DEACTIVATE_TMP="./venv/bin/deactivate.tmp"

# usage function
usage() { echo "Usage: $0 [-m CUSTOM_MARABOU_PATH]" 1>&2; exit 1; }

# parse CLI options
while getopts "hm:" o; do
  case "${o}" in
  m)   PATH_TO_MARABOU="$OPTARG";;
  h|*) usage;;
  esac
done
shift $((OPTIND-1))

# maintain a backup of original in case script is run multiple times.
if [ -f "$_VENV_ACTIVATE_BAK" ]; then
    # backup already exists. restore it before modifying
    cp $_VENV_ACTIVATE_BAK $_VENV_ACTIVATE
else
    # no backup exists, create a copy of the original
    cp $_VENV_ACTIVATE $_VENV_ACTIVATE_BAK
fi

# store new activate code in a temp file
cat > $_VENV_ACTIVATE_TMP <<- EOM

# set custom path variables, and save original values

export _OLD_MARABOU_PATH=\$MARABOU_PATH
export MARABOU_PATH=$PATH_TO_MARABOU
export _OLD_PYTHONPATH=\$PYTHONPATH
export PYTHONPATH=\$MARABOU_PATH:$PROJECT_PATH
export _OLD_JUPYTER_PATH=\$JUPYTER_PATH
export JUPYTER_PATH=\$MARABOU_PATH

# custom marabou command
marabou(){
  \$MARABOU_PATH/build/Marabou "\$@"
}
EOM

# store new deactivate code in a temp file
cat > $_VENV_DEACTIVATE_TMP <<- EOM

# custom deactivate function
deactivate () {
    # unset custom varaibles
    export MARABOU_PATH=\$_OLD_MARABOU_PATH
    unset _OLD_MARABOU_PATH
    export PYTHONPATH=\$_OLD_PYTHONPATH
    unset _OLD_PYTHONPATH
    export JUPYTER_PATH=\$_OLD_JUPYTER_PATH
    unset _OLD_JUPYTER_PATH

    # unset custom commands
    unset -f marabou >/dev/null 2>&1

    # call original deactivate function
    __orig_deactivate__ "\$1"
}

EOM

# rename real 'deactivate' function to __orig_deactivate__
sed -i.dummy "s/^deactivate\ ()\ {/__orig_deactivate__\ ()\ {/" $_VENV_ACTIVATE

# insert deactivate code before "# unset irrelevant variables"
sed -i.dummy '/^#\ unset\ irrelevant\ variables/ {
r ./venv/bin/deactivate.tmp
N
}' $_VENV_ACTIVATE

# insert activate code after "deactivate nondestructive"
sed -i.dummy "/^deactivate\ nondestructive/r ./venv/bin/activate.tmp" $_VENV_ACTIVATE

# cleanup 
rm $_VENV_ACTIVATE_TMP $_VENV_DEACTIVATE_TMP $_VENV_ACTIVATE_DUMMY
