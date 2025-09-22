#!/bin/bash
# Install KnowHunt scheduler as a systemd service

echo "Installing KnowHunt Scheduler Service..."

# Copy service file
sudo cp /home/cordlesssteve/projects/KnowHunt/scripts/knowhunt-scheduler.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable the service (start on boot)
sudo systemctl enable knowhunt-scheduler

echo "✓ KnowHunt scheduler service installed"
echo ""
echo "Usage:"
echo "  Start:   sudo systemctl start knowhunt-scheduler"
echo "  Stop:    sudo systemctl stop knowhunt-scheduler"
echo "  Status:  sudo systemctl status knowhunt-scheduler"
echo "  Logs:    sudo journalctl -u knowhunt-scheduler -f"
echo ""
echo "To start now: sudo systemctl start knowhunt-scheduler"