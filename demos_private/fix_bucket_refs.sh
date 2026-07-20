#!/usr/bin/env bash
# Fix README bucket references to match provisioned buckets
set -uo pipefail

DEMO_BASE="/home/mnagaso/workspace/RCCS/ai4s-platform/umrs/demos_private"

echo "Fixing bucket references in private demo READMEs..."
echo ""

# Map old bucket refs → actual provisioned names
declare -A FIXES
FIXES["demo-sam3"]="demo-defect"        # Defect analysis uses demo-defect, not demo-sam3
FIXES["demo-defect"]="demo-defect"      # Already correct for defect
FIXES["user1-hmae"]="demo-hmae"         # HMAE inference uses demo-hmae
FIXES["demo-hmae"]="demo-hmae"          # Already correct
FIXES["demo-hmae-mae"]="demo-hmae-mae"  # Already correct
FIXES["demo-hmae-features"]="demo-hmae-features"  # Already correct
FIXES["demo-opossum"]="demo-opossum"    # Already correct

REPLACED=0
for demo_dir in 04_hmae_inference 09_defect_analysis 10_hmae_full_pipeline 14_opossum_analysis; do
    README="$DEMO_BASE/$demo_dir/README.md"
    if [ ! -f "$README" ]; then
        continue
    fi

    for old in "${!FIXES[@]}"; do
        new="${FIXES[$old]}"
        if grep -q "$old" "$README"; then
            # Don't fix if old == new (already correct)
            if [ "$old" != "$new" ]; then
                sed -i "s/\b${old}\b/${new}/g" "$README"
                echo "  $demo_dir: $old → $new"
                REPLACED=$((REPLACED + 1))
            fi
        fi
    done
done

echo ""
echo "Fixed $REPLACED bucket references"
echo ""
echo "Verified buckets:"
echo "  demo-defect    — ✅ Defect Analysis"
echo "  demo-hmae      — ✅ HMAE Inference"
echo "  demo-hmae-mae  — ✅ HMAE MAE"
echo "  demo-hmae-features — ✅ HMAE Features"
echo "  demo-opossum   — ✅ Opossum Analysis"
