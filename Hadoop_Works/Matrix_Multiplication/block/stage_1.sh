#!/usr/bin/env bash

hadoop fs -rmr **

hadoop streaming \
-D mapred.job.name="similarity computation : stage 1" \
-D num.key.fields.for.partition=3 \
-D stream.num.map.output.key.fields=3 \
-D mapred.map.tasks=500 \
-D mapred.reduce.tasks=24000 \
-D mapred.job.map.capacity=10000 \
-D mapred.job.reduce.capacity=10000 \
-D mapred.job.priority='HIGH' \
-input "**" \
-output "**" \
-mapper "python26/bin/python26.sh MR1_mapper.py" \
-reducer "python26/bin/python26.sh MR1_reducer.py" \
-file "MR1_mapper.py" \
-file "MR1_reducer.py" \
-cacheArchive "/share/python26.tar.gz#python26"

