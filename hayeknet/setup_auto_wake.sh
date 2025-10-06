#!/bin/bash
# Setup automatic wake and data collection for macOS
# Mountain Time: 8:00 PM daily
# Works even with laptop closed in sleep mode!

set -e

echo "=========================================="
echo "🌙 Setting up Automatic Wake & Collection"
echo "=========================================="
echo ""
echo "⏰ Schedule: 8:00 PM Mountain Time (daily)"
echo "💤 Computer will wake from sleep automatically"
echo "🔋 Works with laptop closed"
echo ""

# Get absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Create logs directory
mkdir -p "$PROJECT_ROOT/logs"

echo "Step 1: Creating Launch Agent for data collection..."
echo ""

# Create LaunchAgent directory if it doesn't exist
mkdir -p ~/Library/LaunchAgents

# Create the Launch Agent plist
cat > ~/Library/LaunchAgents/com.hayeknet.daily.plist << 'PLIST_END'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hayeknet.daily</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>-c</string>
        <string>cd "$PROJECT_ROOT" && source activate_hayeknet.sh && make daily >> logs/daily_$(date +\%Y\%m\%d).log 2>&1</string>
    </array>
    
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>20</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    
    <key>StandardOutPath</key>
    <string>$PROJECT_ROOT/logs/launchd.out</string>
    
    <key>StandardErrorPath</key>
    <string>$PROJECT_ROOT/logs/launchd.err</string>
    
    <key>RunAtLoad</key>
    <false/>
    
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
PLIST_END

echo "✅ Launch Agent created"
echo ""

echo "Step 2: Loading Launch Agent..."
echo ""

# Load the launch agent
launchctl unload ~/Library/LaunchAgents/com.hayeknet.daily.plist 2>/dev/null || true
launchctl load ~/Library/LaunchAgents/com.hayeknet.daily.plist

echo "✅ Launch Agent loaded"
echo ""

echo "Step 3: Setting up automatic wake schedule..."
echo ""
echo "⚠️  This requires administrator password"
echo ""

# Schedule automatic wake at 7:55 PM (5 min before collection)
# This gives computer time to fully wake up
sudo pmset repeat wakeorpoweron MTWRFSU 19:55:00

echo ""
echo "✅ Wake schedule set"
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Configuration:"
echo ""
echo "1. 🌙 Computer wakes: 7:55 PM MT (daily)"
echo "2. 🔄 Data collection runs: 8:00 PM MT"
echo "3. 💤 Computer can sleep again after"
echo ""
echo "=========================================="
echo "🧪 Testing"
echo "=========================================="
echo ""
echo "Test the setup (runs immediately):"
echo "  cd $PROJECT_ROOT && make daily"
echo ""
echo "Check if Launch Agent is loaded:"
echo "  launchctl list | grep hayeknet"
echo ""
echo "View wake schedule:"
echo "  pmset -g sched"
echo ""
echo "Check logs:"
echo "  tail -f $PROJECT_ROOT/logs/launchd.out"
echo ""
echo "=========================================="
echo "⚠️  Important Notes"
echo "=========================================="
echo ""
echo "1. 🔌 Power adapter recommended"
echo "   - Scheduled wake may not work on battery"
echo "   - Connect to power before sleep"
echo ""
echo "2. 🌐 WiFi must reconnect automatically"
echo "   - System Preferences > Network"
echo "   - Make sure auto-join is enabled"
echo ""
echo "3. 🔐 Computer must be logged in"
echo "   - Don't log out, just close laptop"
echo "   - Sleep mode only (not shutdown)"
echo ""
echo "4. 📊 First run: Tomorrow at 8:00 PM MT"
echo "   - Check logs the next morning"
echo "   - Verify data was collected"
echo ""
echo "=========================================="
echo "🛠️  Management Commands"
echo "=========================================="
echo ""
echo "Stop automatic collection:"
echo "  launchctl unload ~/Library/LaunchAgents/com.hayeknet.daily.plist"
echo ""
echo "Start automatic collection:"
echo "  launchctl load ~/Library/LaunchAgents/com.hayeknet.daily.plist"
echo ""
echo "Remove wake schedule:"
echo "  sudo pmset repeat cancel"
echo ""
echo "Restart setup:"
echo "  ./setup_auto_wake.sh"
echo ""
echo "=========================================="
echo ""
echo "✅ Your Mac will now wake and collect data automatically!"
echo "   Close your laptop and let it sleep - it will wake at 8 PM"
echo ""
