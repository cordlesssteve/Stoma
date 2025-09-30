#!/bin/bash
# Install Stoma scheduler as a systemd service

echo "Installing Stoma Scheduler Service..."

# Copy service file
sudo cp /home/cordlesssteve/projects/Stoma/scripts/stoma-scheduler.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Enable the service (start on boot)
sudo systemctl enable stoma-scheduler

echo "✓ Stoma scheduler service installed"
echo ""
echo "Usage:"
echo "  Start:   sudo systemctl start stoma-scheduler"
echo "  Stop:    sudo systemctl stop stoma-scheduler"
echo "  Status:  sudo systemctl status stoma-scheduler"
echo "  Logs:    sudo journalctl -u stoma-scheduler -f"
echo ""
echo "To start now: sudo systemctl start stoma-scheduler"