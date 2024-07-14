#!/bin/bash -ex

echo > stdout.log
memtier_benchmark --random-data --command="pfadd key1 __data__" -n 10000 --hdr-file-prefix "key1_add" >> stdout.log
memtier_benchmark --random-data --command="pfadd key2 __data__" -n 10000 --hdr-file-prefix "key2_add" >> stdout.log
memtier_benchmark --random-data --command="pfadd key3 __data__" -n 10000 --hdr-file-prefix "key3_add" >> stdout.log

memtier_benchmark --command="pfcount key1" -n 10000 --hdr-file-prefix "key1_count" >> stdout.log

memtier_benchmark --command="pfmerge keyall key1 key2 key3" -n 10000 --hdr-file-prefix "key_merge" >> stdout.log

# cleanup
redis-cli del key1 key2 key3 keyall
