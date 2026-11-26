#!/usr/bin/env bash
# Downloads the ESA Kelvins Collision Avoidance Challenge data (public links).
set -e
cd "$(dirname "$0")/.." && mkdir -p data_full/kelvins && cd data_full/kelvins
BASE=https://kelvins.esa.int/media/public/competitions/collision-avoidance-challenge
curl -L -o train_data.zip $BASE/train_data.zip && unzip -o train_data.zip && rm train_data.zip
curl -L -o test_data.csv $BASE/test_data.csv
