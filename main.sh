#!/bin/bash

PROJPATH="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
source .env.sh
cd $PROJPATH

################################################################################
######################### Running the configs in a loop ########################
################################################################################
# #                host      device   rank  size  index        script             config_tree             
# CFGPREFIXLIST=(#"nano4"   "cuda:0"  "0"   "8"   "0.0"     "poisson.py"      "01_poisson/02_mse2d" \
#                "nano4"    "cuda:1"  "0"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d"  \
#                "nano1"    "cuda:0"  "1"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d"  \
#                "nano1"    "cuda:1"  "2"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d"  \
#                "nano7"    "cuda:0"  "3"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d"  \
#                "nano7"    "cuda:1"  "4"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d"  \
#                "nano7"    "cuda:2"  "5"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d"  \
#                "nano7"    "cuda:3"  "6"   "7"   "0.0"     "poisson.py"      "01_poisson/03_ds2d")

#                host      device   rank  size  index       script             config_tree   
CFGPREFIXLIST=("gpub073"  "cuda:0"  "0"   "4"   "0.0"  "smolluchowski.py"   "02_smoll/02_mse" \
               "gpub073"  "cuda:1"  "1"   "4"   "0.0"  "smolluchowski.py"   "02_smoll/02_mse" \
               "gpub073"  "cuda:2"  "2"   "4"   "0.0"  "smolluchowski.py"   "02_smoll/02_mse"
               "gpub073"  "cuda:3"  "3"   "4"   "0.0"  "smolluchowski.py"   "02_smoll/02_mse")

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
  if [ "${myname%%.*}" != ${MYHOST} ]; then
    continue
  fi

  LOGPSTFIX=$(printf "%02d" ${MYRANK})
  OUTLOG="./joblogs/${MYCFGTREE##*/}_${LOGPSTFIX}.out"
  echo "Running Configuration $MYCFGTREE"
  echo "  ==> The json config will be read from ./configs/${MYCFGTREE}.json"
  echo "  ==> The results hdf file will be saved at ./results/${MYCFGTREE}.csv"
  echo "  ==> The training logs will be saved at ${OUTLOG}"
  echo "  + python bspinn/${MYPYSRC} -c ${MYCFGTREE} -d ${MYDEVICE} -s ${MYSIZE} -r ${MYRANK} -i ${MYINDEX} > $OUTLOG 2>&1"
  python bspinn/${MYPYSRC} -c ${MYCFGTREE} -d ${MYDEVICE} -s ${MYSIZE} -r ${MYRANK} -i ${MYINDEX} > $OUTLOG 2>&1 &
  BGPIDS="${BGPIDS} $!"
  echo "----------------------------------------"
done

wait
