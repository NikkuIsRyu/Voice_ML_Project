# Remote Access Guide — Antigravity Chinese Pronunciation Tutor

This guide details how to access your Chinese Pronunciation Tutor running on your GPU PC from your laptop while connected to university Wi-Fi (**eduroam**), a cafe, or a mobile hotspot — with **zero router port forwarding**, end-to-end WireGuard encryption, and full HTTPS support for browser microphone recording.

---

## Why Tailscale is the Recommended Solution for Uni / eduroam

1. **Bypasses Campus Firewalls & CGNAT:** Eduroam and university dorm networks strictly block incoming connections and inbound port forwarding. Tailscale establishes outbound WireGuard tunnels using NAT traversal (DERP relays), connecting your laptop directly to your home desktop.
2. **Built-in HTTPS for Microphone Permissions:** Modern web browsers (Chrome, Edge, Safari) **strictly block microphone access** (`getUserMedia`) on remote HTTP sites unless it is on `localhost` or served over valid **HTTPS**. Tailscale includes automatic Let's Encrypt SSL certificates via **Tailscale Serve / MagicDNS** for free.
3. **Completely Private:** Only your own authenticated devices can see or access the Tutor application. GPT-SoVITS training and WebUI ports (9874, 9880) remain safely bound to `127.0.0.1` and never exposed.

---

## 3-Step Setup Guide

### Step 1: Install Tailscale on Both Machines
1. On your **Desktop (GPU PC)** and **Laptop**, download and install Tailscale from [https://tailscale.com/download](https://tailscale.com/download).
2. Log in with your **Google or GitHub account** on both devices.
3. In your Tailscale Admin Console or tray app, note your Desktop's Tailscale name (e.g. `desktop-gpu`) or Tailscale IP (e.g. `100.x.y.z`).

---

### Step 2: Enable MagicDNS & HTTPS Serve (Desktop)
1. On your desktop, open PowerShell / Command Prompt as Administrator and run:
   ```powershell
   tailscale cert
   ```
2. Enable Tailscale HTTPS Serve to forward port 443 directly to your Tutor service on port 8000:
   ```powershell
   tailscale serve --https=443 8000
   ```
3. Tailscale will output your public secure URL, for example:
   ```
   https://desktop-gpu.tailnet-xyz.ts.net/
   ```

---

### Step 3: Access from Your Laptop at University
1. Turn on Tailscale on your laptop (connected to eduroam).
2. Open Chrome or Edge on your laptop and visit your Tailscale URL:
   ```
   https://desktop-gpu.tailnet-xyz.ts.net/
   ```
3. The browser will grant microphone access immediately because it has a valid SSL certificate.
4. You can now:
   - Practice vocabulary lessons and minimal pairs.
   - Speak into your laptop microphone and receive instant tone scoring + Praat/YIN pitch contour curves.
   - Role-play conversations with the native TTS partner in real-time.

---

## Fallback Option: Direct Tailscale IP (HTTP)
If you prefer not to use `tailscale serve`, you can access directly via:
```
http://100.x.y.z:8000
```
*(Note: In Chrome, to enable microphone permissions on non-HTTPS remote IPs, navigate to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://100.x.y.z:8000`, and restart Chrome).*

---

## Auto-Start on Boot
To ensure your Desktop is always ready when you leave for campus:
- Run `run_tutor.bat` or place a shortcut in your Windows Startup folder (`shell:startup`).
- Verify Tailscale is set to start on Windows boot.
