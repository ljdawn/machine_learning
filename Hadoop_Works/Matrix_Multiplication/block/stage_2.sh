#!/usr/bin/env bash

hadoop dfs -rmr **

hadoop streaming \
-D mapred.job.name="similarity computation : stage 2" \
-D num.key.fields.for.partition=2 \
-D stream.num.map.output.key.fields=2 \
-D mapred.map.tasks=500 \
-D mapred.reduce.tasks=5000 \
-D mapred.job.map.capacity=10000 \
-D mapred.job.reduce.capacity=10000 \
-D mapred.job.priority='HIGH' \
-input "**" \
-output "**" \
-mapper "python26/bin/python26.sh MR2_mappen.py" \
-reducer "python26/bin/python26.sh MR2_reducer.py" \
-file "MR2_mapper.py" \
-file "MR2_reducer.py" \
-cacheArchive "/share/python26.tar.gz#python26"

