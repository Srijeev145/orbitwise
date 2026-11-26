#!/usr/bin/env bash
# Downloads the hand-labelled part of Sen1Floods11 (~700 MB) from its public Google Cloud bucket.
# Needs gsutil (Google Cloud SDK).
set -e
cd "$(dirname "$0")/.." && mkdir -p data_full/sen1floods11 && cd data_full/sen1floods11
B=gs://sen1floods11/v1.1
gsutil -m cp -r $B/data/flood_events/HandLabeled/S1Hand $B/data/flood_events/HandLabeled/LabelHand \
  $B/data/flood_events/HandLabeled/S1OtsuLabelHand $B/data/flood_events/HandLabeled/JRCWaterHand \
  $B/splits/flood_handlabeled .
