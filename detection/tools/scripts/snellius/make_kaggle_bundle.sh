#!/usr/bin/env bash
#
# Pack checkpoints + a small validation subset into one folder to upload as a Kaggle dataset (for the analysis
# notebook detection/tools/analysis/kaggle_analysis.ipynb). Run on a Snellius LOGIN node, from anywhere:
#
#   bash detection/tools/scripts/snellius/make_kaggle_bundle.sh
#
# Settings (environment variables):
#   DATA_ROOT         dataset root                                    (default ~/data/dsert_roll)
#   BUNDLE            output folder                                   (default ~/kaggle_bundle)
#   RUNS              run TAGs whose newest checkpoint to include      (default: livox_pretrain radar_scratch radar_ft_from_livox)
#   SEQS_PER_WEATHER  validation sequences per weather to include      (default 2)
#
# Checkpoints are reduced to their model weights (no optimizer state), named <TAG>.pth. Runs that are still
# training contribute their newest epoch, so you can already practise; re-run the script after training finishes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/snellius_env.sh"
activate_env

DATA_ROOT="${DATA_ROOT:-$HOME/data/dsert_roll}"
BUNDLE="${BUNDLE:-$HOME/kaggle_bundle}"
RUNS="${RUNS:-livox_pretrain radar_scratch radar_ft_from_livox}"
SEQS_PER_WEATHER="${SEQS_PER_WEATHER:-2}"
CKPT_ROOT="$REPO_ROOT/detection/output/det_model_cfgs/dsert/lidar_only"

mkdir -p "$BUNDLE/checkpoints" "$BUNDLE/data/ImageSets/final" "$BUNDLE/data/processed_data"

# --- checkpoints: newest epoch of each run, weights only ------------------------------------------------------
for run in $RUNS; do
    newest=$(ls -t "$CKPT_ROOT/$run/ckpt/"checkpoint_epoch_*.pth 2>/dev/null | head -1 || true)
    if [ -z "$newest" ]; then
        echo "skip $run: no checkpoint yet in $CKPT_ROOT/$run/ckpt"
        continue
    fi
    python - "$newest" "$BUNDLE/checkpoints/$run.pth" <<'EOF'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location='cpu', weights_only=False)
torch.save({'model_state': ckpt['model_state'], 'epoch': ckpt.get('epoch'), 'source': src}, dst)
print('checkpoint %-28s epoch %s  <- %s' % (dst.rsplit('/', 1)[-1], ckpt.get('epoch'), src))
EOF
done

# --- validation subset: first SEQS_PER_WEATHER val sequences of each weather ----------------------------------
python - "$DATA_ROOT/ImageSets/final/val.txt" "$SEQS_PER_WEATHER" "$BUNDLE/data/ImageSets/final/val.txt" <<'EOF'
import sys, collections
src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
per = collections.defaultdict(list)
for line in open(src):
    seq = line.strip()
    if seq:
        per[seq.split('/')[0]].append(seq)
chosen = [s for w in sorted(per) for s in per[w][:n]]
open(dst, 'w').write('\n'.join(chosen) + '\n')
print('val subset: %d sequences (%s)' % (len(chosen), ', '.join('%s:%d' % (w, min(n, len(v))) for w, v in sorted(per.items()))))
EOF
cp "$BUNDLE/data/ImageSets/final/val.txt" "$BUNDLE/data/ImageSets/final/train.txt"   # the dataset class expects both files

while read -r seq; do
    [ -z "$seq" ] && continue
    mkdir -p "$BUNDLE/data/processed_data/$seq"
    rsync -a "$DATA_ROOT/processed_data/$seq/label.pkl" "$DATA_ROOT/processed_data/$seq/LIDAR_LIVOX_Tilted" \
        "$DATA_ROOT/processed_data/$seq/RADAR_Tilted" "$BUNDLE/data/processed_data/$seq/"
done < "$BUNDLE/data/ImageSets/final/val.txt"

cat > "$BUNDLE/dataset-metadata.json" <<EOF
{
  "title": "dsert-roll-analysis-bundle",
  "id": "${KAGGLE_USERNAME:-YOUR_KAGGLE_USERNAME}/dsert-roll-analysis-bundle",
  "licenses": [{"name": "other"}]
}
EOF

echo ""
echo "bundle: $BUNDLE ($(du -sh "$BUNDLE" | cut -f1), $(find "$BUNDLE" -type f | wc -l) files)"
echo "Upload (first time: create; later: new version), from a login node:"
echo "    kaggle datasets create -p $BUNDLE --dir-mode zip"
echo "    kaggle datasets version -p $BUNDLE --dir-mode zip -m \"update checkpoints\""
