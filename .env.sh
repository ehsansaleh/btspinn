#!/bin/bash

################################################################################
########################### ENVIRONMENT Activation #############################
################################################################################
#  Note:
#    Here you can add your own environment activation commands whether you're
#    using conda, virtualenv, etc.
[[ -f venv/bin/activate ]] && source venv/bin/activate

# Note: You do not need to modify the following sections at all.
################################################################################
################################# PYTHONPATH ###################################
################################################################################
# Note:
#   The following two lines are necessary to make the project fall in the python
#   import path without any installations through pip or other invasive actions
#   to the environment. This allows researchers to make modifications and treat
#   the code like a module at the same time.
SCRIPTDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJPARTENTDIR=$(dirname $SCRIPTDIR)
source $SCRIPTDIR/bspinn/bashfuncs.sh
field_append PYTHONPATH $SCRIPTDIR
export PYTHONPATH=$PYTHONPATH