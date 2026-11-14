#!/bin/zsh
# Runs the Cloudflare named tunnel that exposes the Edge API to the Cloud API.
# Install: copy to ~/.securiot/run-tunnel.sh, chmod 700, and save the tunnel
# token in ~/.securiot/tunnel.token (chmod 600).
exec /opt/homebrew/bin/cloudflared tunnel --no-autoupdate run --token "$(cat "$HOME/.securiot/tunnel.token")"
