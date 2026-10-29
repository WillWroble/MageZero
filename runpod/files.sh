# Start command of the files pod (CPU, network volume at /workspace): fetches the repo's data, builds the experiment
# data sets under /workspace/data, then serves /workspace/runs (logs and models) on port 8000. Idempotent: restart
# the pod after a data upload.
mkdir -p /workspace/runs && exec > /workspace/runs/files.log 2>&1
set -x
apt-get update -qq && apt-get install -y -qq git git-lfs python3 python3-pip >/dev/null
pip install -q h5py numpy
python3 -m http.server 8000 --directory /workspace/runs >/dev/null 2>&1 &
(while sleep 30; do du -sh /workspace/repo/.git/lfs > /workspace/runs/lfs-progress.txt; done) &
cd /workspace
[ -d repo ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q -b graph-encoder-test https://github.com/WillWroble/MageZero.git repo
cd repo && rm -f .git/index.lock && GIT_LFS_SKIP_SMUDGE=1 git fetch -q origin graph-encoder-test \
  && GIT_LFS_SKIP_SMUDGE=1 git reset -q --hard origin/graph-encoder-test \
  && git lfs pull --include="experiment_data/Pauper-Faeries/**" && bash runpod/prepare.sh
echo ready
sleep infinity
