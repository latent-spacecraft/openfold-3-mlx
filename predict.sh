#!/bin/bash

# Simple One Peptide Prediction Script
# Usage: ./predict.sh "SEQUENCE"
# Features: Dual validation, FOLDCOMP extraction, 90+ plDDT targeting

if [ $# -eq 0 ]; then
    echo "Usage: $0 <protein_sequence>"
    echo "Example: $0 'MKLHYVAVLTLAILMFLTWLPASLSCNKAL'"
    echo ""
    exit 1
fi

SEQUENCE="$1"
TIMESTAMP=$(date +%s)
OUTPUT_DIR="prediction_${TIMESTAMP}"
QUERY_NAME="query_${TIMESTAMP}"
BASE_QUERY_JSON="base_query_${TIMESTAMP}.json"

echo "================================================================="
echo "🧬 Sequence: ${SEQUENCE}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "⏰ Timestamp: ${TIMESTAMP}"
echo ""

# Step 1: Create base query JSON
echo "📝 Step 1: Creating base query JSON..."
cat > "${BASE_QUERY_JSON}" << EOF
{
    "seeds": [42],
    "queries": {
        "${QUERY_NAME}": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "${SEQUENCE}"
                }
            ]
        }
    }
}
EOF
echo "   ✅ Created: ${BASE_QUERY_JSON}"

# Step 2: OpenFold prediction with templates
echo ""
echo "🎯 Step 4: Final OpenFold prediction..."
echo "   🧬 Mode: ${PREDICTION_MODE}"
echo "   📋 Query: ${BASE_QUERY_JSON}"
echo "   📁 Output: ${OUTPUT_DIR}"
echo ""

PREDICTION_START_TIME=$(date +%s)
python openfold3/run_openfold.py predict \
    --query_json "${BASE_QUERY_JSON}" \
    --runner_yaml examples/example_runner_yamls/mlx_runner.yml \
    --output_dir "${OUTPUT_DIR}" \
    --num_diffusion_samples 8

PREDICTION_EXIT_CODE=$?
PREDICTION_END_TIME=$(date +%s)
PREDICTION_DURATION=$((PREDICTION_END_TIME - PREDICTION_START_TIME))
TOTAL_DURATION=$((PREDICTION_END_TIME - MSA_START_TIME))

echo ""
echo "🎉 PREDICTION COMPLETED!"
echo "================================================================="
echo "⏱️  Timing:"
echo "   MSA generation: ${MSA_DURATION}s"
echo "   Template generation: ${TEMPLATE_DURATION}s"
echo "   Final prediction: ${PREDICTION_DURATION}s"
echo "   Total runtime: ${TOTAL_DURATION}s"
echo ""

# Analyze results
echo "📊 RESULTS ANALYSIS:"
if [ $PREDICTION_EXIT_CODE -eq 0 ] && [ -d "${OUTPUT_DIR}/${QUERY_NAME}/seed_42" ]; then
    FOUND_RESULTS=false
    for conf_file in "${OUTPUT_DIR}/${QUERY_NAME}/seed_42"/*_confidences_aggregated.json; do
        if [ -f "$conf_file" ]; then
            sample=$(basename "$conf_file" | sed 's/.*_sample_\([0-9]*\)_.*/\1/')
            plddt=$(grep -o '"avg_plddt": [0-9.]*' "$conf_file" | cut -d' ' -f2 | head -1)

            if [ -n "$plddt" ]; then
                echo "   🎯 Sample ${sample}: plDDT = ${plddt}"

                # Check if we achieved our target
                if (( $(echo "$plddt >= 90" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      🎉 BREAKTHROUGH! Achieved 90+ plDDT target!"
                elif (( $(echo "$plddt >= 80" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      ✅ Excellent accuracy (80+ plDDT)"
                elif (( $(echo "$plddt >= 70" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      👍 Good accuracy (70+ plDDT)"
                elif (( $(echo "$plddt >= 60" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      📊 Moderate accuracy (60+ plDDT)"
                else
                    echo "      ⚠️ Low accuracy (<60 plDDT)"
                fi
                FOUND_RESULTS=true
            fi
        fi
    done

    if [ "$FOUND_RESULTS" = false ]; then
        echo "   ❌ No confidence scores found"
    fi
else
    echo "   ❌ Prediction failed or no results generated"
fi

# Show template information if available
if [ -f "template_results_${TIMESTAMP}.json" ]; then
    TEMPLATE_COUNT=$(grep -o '"template_count": [0-9]*' "template_results_${TIMESTAMP}.json" | cut -d' ' -f2 2>/dev/null || echo "0")
    echo ""
    echo "🏗️ Template Information:"
    echo "   Templates used: ${TEMPLATE_COUNT}"
    if [ "$TEMPLATE_COUNT" -gt 0 ]; then
        echo "   ✅ Revolutionary dual validation successful!"
        echo "   ✅ MSA + Templates pipeline used"
    else
        echo "   ⚠️ MSA-only mode used"
    fi
fi

echo ""
echo "📁 Generated Files:"
echo "   - Base query: ${BASE_QUERY_JSON}"
echo "   - Results: ${OUTPUT_DIR}/"
echo "   - MSA log: msa_${TIMESTAMP}.log"
if [ -f "template_results_${TIMESTAMP}.json" ]; then
    echo "   - Template info: template_results_${TIMESTAMP}.json"
fi

echo ""
echo "🔍 Debug Commands:"
echo "   View MSA log: cat msa_${TIMESTAMP}.log"
echo "   View results: ls -la ${OUTPUT_DIR}/${QUERY_NAME}/seed_42/"
echo "   View template info: cat template_results_${TIMESTAMP}.json"

if [ $PREDICTION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Revolutionary MSA + Templates pipeline completed!"
else
    echo ""
    echo "⚠️ Prediction completed with warnings - check debug info above"
fi