# Reports the volume's size, use and writability, the biggest directories, and the gen5 shards and results (read-only
# apart from this run's own output in $OUT).
df -h /workspace
date > $OUT/write-test.txt && echo "write ok" || echo "WRITE FAILED"
du -xsh /workspace/* /workspace/runs/* 2>/dev/null | sort -h | tail -15
ls -la --time-style=+%H:%M /workspace/runs/gen5-offline/PlayerA | tail -n +2
ls /workspace/runs/gen5-offline/PlayerA | wc -l
for d in /workspace/runs/gen5off-*; do echo "$d $(ls $d | grep result | tr '\n' ' ') DONE=$(cat $d/DONE 2>/dev/null)"; done
