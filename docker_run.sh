#!/bin/bash

docker run \
    --rm \
    --gpus "device=0" \
    --name run_trial \
    -v /home/ricardo/ABUS2023_documents/tdsc_abus23/input:/input/ \
    -v /home/ricardo/ABUS2023_documents/tdsc_abus23/exit:/predict/ \
    --shm-size 8g \
    vicorob:almost