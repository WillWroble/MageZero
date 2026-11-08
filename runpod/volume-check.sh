# Reports the volume's size, use and writability, and the biggest directories (read-only apart from this run's own
# output in $OUT).
df -h /workspace
df -i /workspace
date > $OUT/write-test.txt && echo "write ok" || echo "WRITE FAILED"
du -xsh /workspace/* /workspace/repo/.git/lfs /workspace/runs/* 2>/dev/null | sort -h | tail -30
