#!/bin/bash
# Monitor Pipeline Option A Progress
# Run: bash monitor_pipeline.sh

LOG_FILE="pipeline_optA_full.log"

echo "========================================="
echo "PIPELINE OPTION A MONITOR"
echo "========================================="
echo ""

# Check if pipeline is running
if ps aux | grep -q "[p]ython.*run_multiple_models_pipeline_OPTION_A"; then
    echo "Status: RUNNING"
else
    echo "Status: NOT RUNNING or COMPLETED"
fi

echo ""
echo "Latest Log (last 50 lines):"
echo "-----------------------------------------"
tail -50 "$LOG_FILE" 2>/dev/null || echo "Log file not found"

echo ""
echo "-----------------------------------------"
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo "Experiment Folders:"
ls -lh results/ | grep optA | tail -5

echo ""
echo "To monitor continuously: tail -f $LOG_FILE"
echo "========================================="
