#!/bin/bash
# setup_server.sh — One-command Digital Ocean setup
# Run: bash setup_server.sh

set -e

echo "======================================================"
echo "  ATP Tennis Model — Digital Ocean Setup"
echo "======================================================"

# 1. System packages
apt-get update -q
apt-get install -y python3-pip python3-venv git cron

# 2. Python dependencies
pip3 install -q pandas numpy scipy scikit-learn xgboost requests streamlit

# 3. Environment variables — edit these before running
cat >> ~/.bashrc << 'EOF'

# ATP Tennis Model
export ODDS_API_KEY="a5a8d62fefaa1ae39cdddd9aa19d5f44"
export TELEGRAM_TOKEN=""        # fill in after creating bot
export TELEGRAM_CHAT_ID=""      # fill in after creating bot
export BANKROLL="1000"
EOF
source ~/.bashrc

# 4. Create data directory
mkdir -p /root/backtest_match/tennis/data

# 5. Set up cron job — runs every day at 9:00 AM UTC
(crontab -l 2>/dev/null; echo "0 9 * * * cd /root/backtest_match && python3 tennis/scheduler.py >> tennis/data/scheduler.log 2>&1") | crontab -

echo ""
echo "  Cron job set: runs daily at 9:00 AM UTC"
echo ""

# 6. Start dashboard as background service
cat > /etc/systemd/system/tennis-dashboard.service << 'EOF'
[Unit]
Description=ATP Tennis Dashboard
After=network.target

[Service]
User=root
WorkingDirectory=/root/backtest_match
ExecStart=/usr/local/bin/streamlit run tennis/dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable tennis-dashboard
systemctl start tennis-dashboard

echo ""
echo "======================================================"
echo "  Setup complete!"
echo ""
echo "  Dashboard:  http://YOUR_DROPLET_IP:8501"
echo "  Scheduler:  runs daily at 9am UTC automatically"
echo "  Logs:       tail -f tennis/data/scheduler.log"
echo ""
echo "  Next steps:"
echo "  1. Add TELEGRAM_TOKEN and TELEGRAM_CHAT_ID to ~/.bashrc"
echo "  2. Open port 8501 in your Digital Ocean firewall"
echo "  3. Run: python3 tennis/scheduler.py  (test run now)"
echo "======================================================"
