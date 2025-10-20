#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${HOME}/data"
DST="${DATA_ROOT}/Fruit-Images-Dataset"

mkdir -p "$DATA_ROOT"
if [ -d "$DST/.git" ] || [ -d "$DST/Training" ]; then
  echo "Dataset already present at: $DST"
else
  git clone https://github.com/Horea94/Fruit-Images-Dataset.git "$DST"
fi

echo "Dataset at: $DST"

