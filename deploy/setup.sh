#!/usr/bin/env bash
# One-time setup of the sylearn AQI web app on a fresh Ubuntu 24.04 Vultr server.
# Run as root:  bash setup.sh
# Re-running it is safe: it pulls the latest code and restarts the app.
set -euo pipefail

REPO=https://github.com/Rozerxshashank/AQI-analysis-and-prediction.git
APP_USER=aqi
APP_DIR=/home/$APP_USER/AQI-analysis-and-prediction

echo "==> 1/6 System packages"
apt-get update -y
apt-get install -y git nginx curl ufw

echo "==> 2/6 App user"
id -u $APP_USER >/dev/null 2>&1 || adduser --disabled-password --gecos "" $APP_USER

echo "==> 3/6 Code"
if [ -d "$APP_DIR/.git" ]; then
  sudo -u $APP_USER git -C "$APP_DIR" pull --ff-only
else
  sudo -u $APP_USER git clone "$REPO" "$APP_DIR"
fi

echo "==> 4/6 Python 3.11 + packages (same versions that trained the models)"
sudo -u $APP_USER bash -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
UV=/home/$APP_USER/.local/bin/uv
sudo -u $APP_USER bash -c "cd $APP_DIR && $UV python install 3.11 && $UV venv --python 3.11 --allow-existing .venv && $UV pip install --python .venv/bin/python -r web/requirements.txt"

echo "==> 5/6 Service (auto-start, auto-restart)"
cp "$APP_DIR/deploy/aqi-web.service" /etc/systemd/system/aqi-web.service
systemctl daemon-reload
systemctl enable aqi-web
systemctl restart aqi-web

echo "==> 6/6 nginx on port 80 + firewall"
cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/aqi-web
ln -sf /etc/nginx/sites-available/aqi-web /etc/nginx/sites-enabled/aqi-web
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx
ufw allow OpenSSH && ufw allow 'Nginx Full' && ufw --force enable

echo "==> Waiting for the app to start (about 10 s)..."
for i in $(seq 1 30); do
  if curl -fs http://127.0.0.1:8000/api/meta >/dev/null; then
    echo "Done. Open: http://$(curl -fs4 https://ifconfig.me || hostname -I | awk '{print $1}')/"
    exit 0
  fi
  sleep 2
done
echo "The app did not start. See the log with: journalctl -u aqi-web -n 50"
exit 1
