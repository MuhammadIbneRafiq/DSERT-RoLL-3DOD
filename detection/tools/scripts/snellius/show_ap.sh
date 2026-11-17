#!/bin/bash
# Epoch-20 results of the main runs, read from their eval logs.
# DSERT-RoLL: one row per run, Vehicle Level-1 AP overall and per weather.
# TruckScenes: the devkit summary lines (mAP, NDS, per-class AP).
# Usage: bash detection/tools/scripts/snellius/show_ap.sh
cd ~/DSERT-RoLL-3DOD/detection/output/det_model_cfgs || exit 1

# newest log in a run folder that contains the epoch-20 evaluation
log_of() { grep -rlsE "EPOCH 20 EVALUATION|epoch_20" "$1" --include="*.txt" | xargs -r ls -t 2>/dev/null | head -1; }

printf "%-34s %7s %7s %7s %7s %7s %7s %7s\n" RUN overall Clear Fog H_Rain H_Snow L_Rain L_Snow
for d in dsert/lidar_only/* dsert/voxelnext/vnx_*_s[123]; do
    case "$d" in *smoke*|*bad_heatmap*|*radar_e20_on_livox) continue;; esac
    f=$(log_of "$d"); [ -n "$f" ] || { printf "%-34s no log\n" "${d##*/}"; continue; }
    awk -v run="${d##*/}" '
        /EPOCH 20 EVALUATION/ { on=1; delete ap; title="" }
        on && /Per Light/ { on=0 }
        on && match($0, /\[(overall|weather=[^]]+)\] N=/) { title=substr($0, RSTART+1, RLENGTH-5); sub(/weather=/, "", title) }
        on && title != "" && /VEHICLE_LEVEL_1\/AP:/ { ap[title]=$NF }
        END {
            n=split("overall Clear Fog Heavy_Rain Heavy_Snow Light_Rain Light_Snow", k, " ")
            printf "%-34s", run
            for (i=1; i<=n; i++) printf " %7s", (k[i] in ap) ? ap[k[i]] : "-"
            printf "\n"
        }' "$f"
done

for d in truckscenes/voxelnext/vnx_ts_lidar truckscenes/voxelnext/vnx_ts_radar truckscenes/voxelnext/vnx_ts_radar_ft_from_lidar; do
    echo; echo "== ${d##*/}"
    f=$(log_of "$d"); [ -n "$f" ] || { echo "no log"; continue; }
    awk '/EPOCH 20 EVALUATION/ { buf=""; on=1 } on { buf = buf $0 "\n" } END { printf "%s", buf }' "$f" \
        | grep -iE "mAP|NDS|^ *(car|truck|bus|trailer|other_vehicle|pedestrian|bicycle|motorcycle)|Object Class" | tail -25
done
