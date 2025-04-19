#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source ./activate venv
cd $PROJPATH

################################################################################
######################### Running the configs in a loop ########################
################################################################################
#              hostname   device   rank  size  index     script             config_tree      
CFGPREFIXLIST=(   '*'    "cuda:0"  "0"   "1"   "0.0"  "poisson.py"   "01_poisson/30_btstrp2d")

# Uncomment the next lines if you want to split the work among four gpus.
# #              hostname   device   rank  size  index     script             config_tree      
# CFGPREFIXLIST=(   '*'    "cuda:0"  "0"   "4"   "0.0"  "poisson.py"   "01_poisson/30_btstrp2d" \
#                   '*'    "cuda:1"  "1"   "4"   "0.0"  "poisson.py"   "01_poisson/30_btstrp2d" \
#                   '*'    "cuda:2"  "2"   "4"   "0.0"  "poisson.py"   "01_poisson/30_btstrp2d" \
#                   '*'    "cuda:3"  "3"   "4"   "0.0"  "poisson.py"   "01_poisson/30_btstrp2d")

BGPIDS=""
trap 'kill $BGPIDS; exit' INT
mkdir -p joblogs
for (( i=0; i<${#CFGPREFIXLIST[*]}; i+=7)); do
  MYHOST="${CFGPREFIXLIST[$((i+0))]}"
  MYDEVICE="${CFGPREFIXLIST[$((i+1))]}"
  MYRANK="${CFGPREFIXLIST[$((i+2))]}"
  MYSIZE="${CFGPREFIXLIST[$((i+3))]}"
  MYINDEX="${CFGPREFIXLIST[$((i+4))]}"
  MYPYSRC="${CFGPREFIXLIST[$((i+5))]}"
  MYCFGTREE="${CFGPREFIXLIST[$((i+6))]}"

  myname=$(hostname)
  if [[ "${myname%%.*}" != ${MYHOST} ]] ; then
    continue
  fi

  LOGPSTFIX=$(printf "%02d" ${MYRANK})
  OUTLOG="./joblogs/${MYCFGTREE##*/}_${LOGPSTFIX}.out"
  echo "Running Configuration $MYCFGTREE"
  echo "  ==> The yaml config will be read from ./configs/${MYCFGTREE}.yml"
  echo "  ==> The results hdf file will be saved at ./results/${MYCFGTREE}.h5"
  echo "  ==> The training logs will be saved at ${OUTLOG}"
  echo "  + python bspinn/${MYPYSRC} -c ${MYCFGTREE} -d ${MYDEVICE} -s ${MYSIZE} -r ${MYRANK} -i ${MYINDEX} > $OUTLOG 2>&1"
  python bspinn/${MYPYSRC} -c ${MYCFGTREE} -d ${MYDEVICE} -s ${MYSIZE} -r ${MYRANK} -i ${MYINDEX} > $OUTLOG 2>&1 &
  BGPIDS="${BGPIDS} $!"
  echo "----------------------------------------"
done

wait
