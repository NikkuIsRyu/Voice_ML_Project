# Cloudflare Tunnel & Free Custom Domain Guide

This guide explains how to get a **100% free custom domain** via the **GitHub Student Developer Pack** and connect it to your GPU Desktop using **Cloudflare Zero Trust Tunnels (`cloudflared`)**.

---

## Benefits of Cloudflare Tunnel
- **Zero Port Forwarding:** Works behind any router/NAT; no public IP required.
- **Free Automated SSL:** Valid HTTPS everywhere (critical for browser microphone access).
- **Cloudflare Access Authentication:** You can add an email PIN code login or Google OAuth in front of your domain so random internet users cannot access your tutor.
- **Custom Domain:** Access your tutor via your own clean domain (e.g., `https://chinese.yourname.me`).

---

## Step 1: Claim Free Domain with GitHub Student Pack
1. Apply or log into [GitHub Student Developer Pack](https://education.github.com/pack).
2. Claim the free domain offer:
   - **Namecheap:** 1 free `.me` domain registration for 1 year with free WhoisGuard.
   - **Name.com:** 1 free `.live`, `.tech`, or `.me` domain for 1 year.
3. Complete registration for your preferred domain name (e.g. `yourname.me`).

---

## Step 2: Add Domain to Cloudflare (Free Plan)
1. Create a free account at [Cloudflare.com](https://cloudflare.com).
2. Click **Add a site** and enter your new domain (`yourname.me`). Choose the **Free Plan**.
3. Cloudflare will give you two nameservers (e.g., `alice.ns.cloudflare.com`, `bob.ns.cloudflare.com`).
4. In your Namecheap / Name.com control panel, change your domain's DNS Nameservers to **Custom DNS** and paste the Cloudflare nameservers.
5. Wait 2-5 minutes for Cloudflare to verify the domain.

---

## Step 3: Install & Configure Cloudflare Tunnel on Your Desktop
1. In Cloudflare, navigate to the **Zero Trust Dashboard** -> **Networks** -> **Tunnels**.
2. Click **Add a tunnel** and select **Cloudflared**. Name it `voice-gpu-tunnel`.
3. Choose **Windows** and download the `cloudflared.exe` installer or run the provided command in an Administrator PowerShell:
   ```powershell
   winget install --id Cloudflare.cloudflared
   ```
4. Run the token connector command shown on screen:
   ```powershell
   cloudflared service install <YOUR_TUNNEL_TOKEN>
   ```
5. In the Cloudflare Tunnel Web UI, configure a **Public Hostname**:
   - **Subdomain:** `tutor` (or leave blank for root domain)
   - **Domain:** `yourname.me`
   - **Service Type:** `HTTP`
   - **URL:** `localhost:8000`
6. Click **Save Hostname**.

---

## Step 4: Protect with Cloudflare Access (Optional but Recommended)
To prevent unauthorized visitors from consuming your GPU/CPU:
1. In Cloudflare Zero Trust, go to **Access** -> **Applications** -> **Add an application** -> **Self-hosted**.
2. Set Application Domain: `tutor.yourname.me`.
3. Under **Policies**, create an "Allow" rule:
   - **Rule action:** Allow
   - **Include:** Selector -> *Emails* -> Enter your personal email address.
4. Now, visiting `https://tutor.yourname.me` prompts for a one-time code sent to your email before opening the tutor app!
