#!/bin/bash
# Setup script for automated daily ERCOT data collection
# This script configures cron to run the daily data collector automatically

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "================================================================================"
echo "ERCOT Daily Data Collection - Setup"
echo "================================================================================"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Check if hayeknet environment exists
if ! conda env list | grep -q "hayeknet"; then
    echo "❌ hayeknet conda environment not found."
    echo "   Please create it first with: conda env create -f environment.yml"
    exit 1
fi

echo "✅ Environment checks passed"
echo ""

# Get conda path
CONDA_BASE=$(conda info --base)
CONDA_ACTIVATE="$CONDA_BASE/etc/profile.d/conda.sh"

# Create wrapper script that activates conda
WRAPPER_SCRIPT="$SCRIPT_DIR/run_daily_collection.sh"

cat > "$WRAPPER_SCRIPT" << 'EOF'
#!/bin/bash
# Wrapper script to activate conda and run daily collection

# Source conda
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "Error: Cannot find conda activation script"
    exit 1
fi

# Activate hayeknet environment
conda activate hayeknet

# Change to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_DIR"

# Run daily collector
python scripts/daily_data_collector.py >> data/logs/daily_collection.log 2>&1

# Exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Daily collection successful" >> data/logs/daily_collection.log
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ❌ Daily collection failed with code $EXIT_CODE" >> data/logs/daily_collection.log
fi

exit $EXIT_CODE
EOF

chmod +x "$WRAPPER_SCRIPT"

echo "✅ Created wrapper script: $WRAPPER_SCRIPT"
echo ""

# Determine what time to run (2 AM daily)
CRON_TIME="0 2 * * *"  # 2 AM every day
CRON_JOB="$CRON_TIME $WRAPPER_SCRIPT"

echo "📅 Proposed cron schedule:"
echo "   Time: 2:00 AM daily"
echo "   Command: $CRON_JOB"
echo ""

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_data_collector.py\|run_daily_collection.sh"; then
    echo "⚠️  Existing cron job found for daily data collection"
    echo ""
    echo "Current crontab:"
    crontab -l 2>/dev/null | grep -E "daily_data_collector|run_daily_collection" || true
    echo ""
    read -p "Remove existing job and install new one? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove old jobs
        crontab -l 2>/dev/null | grep -v -E "daily_data_collector|run_daily_collection" | crontab -
        echo "✅ Removed old cron job"
    else
        echo "❌ Setup cancelled. Manual configuration required."
        exit 0
    fi
fi

# Add new cron job
echo "📝 Installing cron job..."
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✅ Cron job installed successfully!"
echo ""

# Verify installation
echo "📋 Installed cron jobs:"
crontab -l 2>/dev/null | grep "run_daily_collection.sh" || echo "   (none found)"
echo ""

# Create log directory
mkdir -p "$PROJECT_DIR/data/logs"

echo "================================================================================"
echo "✅ Setup Complete!"
echo "================================================================================"
echo ""
echo "📊 Daily data collection is now automated:"
echo "   • Runs at: 2:00 AM every day"
echo "   • Collects: Last 24 hours of ERCOT LMP data"
echo "   • Appends to: data/archive/ercot_lmp/"
echo "   • Generates: Daily analysis report in data/reports/"
echo "   • Logs to: data/logs/daily_collection.log"
echo ""
echo "🔍 Monitor execution:"
echo "   tail -f $PROJECT_DIR/data/logs/daily_collection.log"
echo ""
echo "🧪 Test now (without waiting for cron):"
echo "   $WRAPPER_SCRIPT"
echo ""
echo "📝 View cron jobs:"
echo "   crontab -l"
echo ""
echo "🗑️  Remove automation:"
echo "   crontab -e  # Then delete the line with run_daily_collection.sh"
echo ""
echo "================================================================================"
