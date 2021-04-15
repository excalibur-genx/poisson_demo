#!/bin/bash
# :mem=500gb for 512GB nodes

BATCH_NAME=poisson_multinode_patchpc_x
SOLVER_PARAMS="MG F-cycle PatchPC telescope"
RESULTS_DIR=results/${BATCH_NAME}
mkdir -p ${RESULTS_DIR}

SPACING=1
for NODES in 1 2 4 8 #16 32 64
    do
    if [ $NODES -eq 1 ]
        then
        SPECIAL=":mem=500gb"
    else
        unset SPECIAL
    fi
    echo ${NODES}${SPECIAL}
    qsub -l select=${NODES}${SPECIAL} \
        -N ${BATCH_NAME} \
        -o ${RESULTS_DIR}/${NODES}_nodes.out \
        -e ${RESULTS_DIR}/${NODES}_nodes.err \
        -v NODES=${NODES},SPACING=${SPACING},RESULTS_DIR=${RESULTS_DIR},SOLVER_PARAMS="${SOLVER_PARAMS}" \
        poisson/poisson.pbs
done
