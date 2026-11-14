#!/bin/bash
#
# Download MAN TruckScenes v1.2 to scratch, check each zip against the published MD5, unzip only the sensors you
# keep, then delete the zip. At most PARALLEL zips (~80 GB each) sit on disk at once.
#
#   sbatch detection/tools/scripts/snellius/download_truckscenes.sbatch                      # trainval, lidar+radar
#   sbatch --export=ALL,KEEP=all detection/tools/scripts/snellius/download_truckscenes.sbatch # also cameras
#   sbatch --export=ALL,SPLITS="trainval test" ...                                             # also test (NO labels)
#
# Extracted sizes (read from the zips' file index, 2026-11-14):
#   trainval  lidar 459 GB (samples 91 + sweeps 368)  radar 19 GB (2 + 17)  camera 186 GB  metadata 4.3 GB
#   test      lidar 116 GB                            radar  5 GB           camera  49 GB  metadata 0.9 GB
# The test split ships an empty sample_annotation.json, so it can only be scored on the TruckScenes server.
# Train/val both come from the trainval zips; the devkit's splits define which scenes are val.
#
# Re-submitting is safe: finished zips are skipped (marker files in $DEST/.done), partial downloads resume.
# /scratch-shared deletes files after 14 days. Check space first with `myquota`.
# File counts matter too (scratch inode quota is 3M): trainval lidar 0.72M + radar 1.38M files = 2.1M;
# adding cameras (+0.6M) and test (+0.65M) gives 3.2M and goes over that quota.
#
#SBATCH --job-name=truckscenes-dl
#SBATCH --partition=staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

DEST="${DEST:-/scratch-shared/$USER/truckscenes}"
SPLITS="${SPLITS:-trainval}"
KEEP="${KEEP:-LIDAR RADAR}"        # sensor-name prefixes to extract, or "all"
PARALLEL="${PARALLEL:-3}"
BASE=https://man-truckscenes.s3.eu-central-1.amazonaws.com/release

# name md5 (from the TruckScenes download page)
read -r -d '' TABLE <<'EOF' || true
trainval/man-truckscenes_metadata_v1.2-trainval.zip     dd35002927eef8f8c49267783c317306
trainval/man-truckscenes_sensordata01_v1.2-trainval.zip 95b83a96110269296f663dddf683d10c
trainval/man-truckscenes_sensordata02_v1.2-trainval.zip fd6b60c8cea64947ad139421bcc4134b
trainval/man-truckscenes_sensordata03_v1.2-trainval.zip 130856e0fd88124033d390ac6f3204b2
trainval/man-truckscenes_sensordata04_v1.2-trainval.zip 1eb598bf2a952dd401a23ee1d08a827b
trainval/man-truckscenes_sensordata05_v1.2-trainval.zip 5c74205ae352a225c5d8b16d40a7d18a
trainval/man-truckscenes_sensordata06_v1.2-trainval.zip 1b27a832f6a14ebfd70149c1ec8e7f70
trainval/man-truckscenes_sensordata07_v1.2-trainval.zip c1f2b4f2f7e7a5247b3dbcac148e0278
test/man-truckscenes_metadata_v1.2-test.zip             e789d70bd1051fed14990f1fccc947dd
test/man-truckscenes_sensordata01_v1.2-test.zip         368d8656e162a69ba10f33807c84b756
test/man-truckscenes_sensordata02_v1.2-test.zip         a62d1a2972c255eaea1bf7942a23354a
EOF

mkdir -p "$DEST/zips" "$DEST/.done"
export DEST KEEP BASE

fetch() {
    local rel="$1" md5="$2" zip name
    name="$(basename "$rel")"; zip="$DEST/zips/$name"
    if [[ -e "$DEST/.done/$name" ]]; then echo "skip $name (done)"; return 0; fi

    echo "[$(date +%T)] download $name"
    wget -q -c --tries=20 --waitretry=30 --no-use-server-timestamps -O "$zip" "$BASE/$rel"

    echo "[$(date +%T)] md5 $name"
    if [[ "$(md5sum "$zip" | cut -d' ' -f1)" != "$md5" ]]; then
        echo "MD5 MISMATCH $name; deleting it, resubmit to retry" >&2; rm -f "$zip"; return 1
    fi

    echo "[$(date +%T)] unzip $name"
    local pats=()
    if [[ "$KEEP" != all && "$name" == *sensordata* ]]; then
        # samples|sweeps/<SENSOR>/<file>; spelled out so it works whether or not '*' crosses '/' in this unzip
        for s in $KEEP; do pats+=("man-truckscenes/*/${s}_*/*"); done
    fi
    # -DD: stamp files with the extraction time, not the archive's months-old dates, so the scratch
    # 14-day cleanup doesn't treat freshly unpacked data as already expired.
    unzip -q -o -DD "$zip" "${pats[@]}" -d "$DEST"

    rm -f "$zip"; touch "$DEST/.done/$name"
    echo "[$(date +%T)] done $name"
}
export -f fetch

for split in $SPLITS; do grep "^$split/" <<<"$TABLE"; done \
    | xargs -P "$PARALLEL" -L 1 bash -c 'fetch "$0" "$1"'

R="$DEST/man-truckscenes"
echo; echo "Per folder:"
du -sh "$R"/samples/* "$R"/sweeps/* "$R"/v1.2-* 2>/dev/null || true
echo; echo "Totals (size, files):"
for s in LIDAR RADAR CAMERA; do
    for part in samples sweeps; do
        compgen -G "$R/$part/${s}_*" >/dev/null || continue
        printf '  %-6s %-7s %8s  %9d files\n' "$s" "$part" \
            "$(du -shc "$R/$part/${s}_"* | tail -1 | cut -f1)" "$(find "$R/$part/${s}_"* -type f | wc -l)"
    done
done
printf '  %-14s %8s  %9d files\n' all "$(du -sh "$R" | cut -f1)" "$(find "$R" -type f | wc -l)"
