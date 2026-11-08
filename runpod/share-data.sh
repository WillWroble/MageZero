# Serves the volume's repo data through the files pod (/workspace/runs is what it serves) and lists every shard with
# its size and SHA-256, for copying the data elsewhere. Undo: rm /workspace/runs/experiment_data
set -e
ln -sfn /workspace/repo/experiment_data /workspace/runs/experiment_data
cd /workspace/repo && find experiment_data/Pauper-Faeries/ver4 experiment_data/Pauper-Faeries/ver5 -name "*.hdf5" -printf "%s %p\n" | sort -k2 > $OUT/files.txt
wc -l $OUT/files.txt
