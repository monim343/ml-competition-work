#!/usr/bin/env bash
# Pulls the one artifact kept out of the tree: the gradient-boosted tree model bundle
# (171 MB, one file 96.9 MB). Everything else the kernel reads is already in weights/.
#
# Requires the Kaggle CLI, authenticated:  pip install kaggle  &&  ~/.kaggle/kaggle.json
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/weights/maek"
mkdir -p "$DEST"
echo "fetching mooniim/maek-ms-model -> $DEST"
kaggle datasets download mooniim/maek-ms-model -p "$DEST" --unzip -q -o
echo
echo "expected contents:"
echo "  models/fold[0-4]_s0.pkl   features.json   manifest.json"
echo
found=$(find "$DEST" -name 'fold*_s0.pkl' | wc -l | tr -d ' ')
echo "fold models found: $found (expect 5)"
if [ -f "$DEST/manifest.json" ]; then
  python3 - "$DEST/manifest.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); oof=m.get('oof_rmse')
print(f"manifest oof_rmse: {oof}")
print("MATCHES the clean trainer's asserted value" if oof and abs(float(oof)-7.895818)<1e-5
      else "DOES NOT match 7.895818 - the bundle is not the one this solution shipped")
PY
fi
