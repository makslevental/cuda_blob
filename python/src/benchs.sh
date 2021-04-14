#set -e
#
#MAX_GPU=$1
#MAX_SIGMA=30
#RESIZE=1024
#
#for GPUS in $(seq 1 $MAX_GPU); do
#  for N_SIGMA_BINS in {2..49}; do
#    a=$((GPUS * N_SIGMA_BINS))
#    if [ $a -gt 48 ]; then
#      continue
#    fi
#
#    for _ in {1..10}; do
#      echo "${GPUS} ${a} ${RESIZE} ${MAX_SIGMA}"
#      ./mpi_run.sh $GPUS $a $RESIZE $MAX_SIGMA >> "logs/${GPUS}_${a}.log"
#    done
#  done
#done

set -e

MAX_GPU=$1
MAX_SIGMA=30

for GPUS in $(seq 1 $MAX_GPU); do
  for N_SIGMA_BINS in {32..32}; do
#  for N_SIGMA_BINS in {2..49}; do
    a=$((GPUS * N_SIGMA_BINS))
    if [ $a -gt 48 ]; then
      continue
    fi

    for R in {8..13}; do
      RESIZE=$((2 ** R))
      for _ in {1..10}; do
        echo "${GPUS} ${a} ${RESIZE} ${MAX_SIGMA}"
        ./mpi_run.sh $GPUS $a $RESIZE $MAX_SIGMA >> "logs/${GPUS}_${a}_${RESIZE}_${MAX_SIGMA}.log"
      done
    done
  done
done
