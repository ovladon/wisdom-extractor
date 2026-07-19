# Hosting the Wisdom Lab on a dedicated Android phone

A spare phone (tested target: ASUS ROG Phone 2 — Snapdragon 855, 8–12 GB RAM) makes an
excellent dedicated server: ~3 W power draw (≈ €0.5/month of electricity vs €10+ for a
desktop), physically isolated from your personal machines, and disposable by design.
No root, no custom ROM — everything runs in Termux + a proot Ubuntu.

## 0. Safety preparation (do this first)

- **Factory-reset the phone.** Skip Google sign-in entirely if it lets you; if not, use
  a fresh throwaway account. The server phone should hold nothing personal.
- Connect it to home Wi-Fi. Remove the SIM.
- **Battery care:** in ASUS's Battery settings / Armoury Crate, enable the charging
  limit (80%) — the phone will live on the charger, and Li-ion batteries kept at 100%
  and warm age badly. Use the original charger; keep the phone ventilated, not under a
  pillow of dust.
- In Settings → Display: screen timeout short; no lock-screen secrets needed.

## 1. Install Termux (from F-Droid, NOT Play Store — Play build is abandoned)

1. In the phone's browser: **f-droid.org** → download the F-Droid APK → allow
   "install unknown apps" when prompted → install.
2. In F-Droid, install **Termux** and **Termux:Boot**.
3. Open Termux once; open Termux:Boot once (that registers it). In Android Settings →
   Apps → Termux → Battery: **Unrestricted / disable battery optimization**.

## 2. Base system (paste into Termux)

```bash
termux-wake-lock
pkg update -y && pkg install -y proot-distro
proot-distro install ubuntu
proot-distro login ubuntu
```

You are now in Ubuntu on the phone.

## 3. One-command setup (inside the Ubuntu prompt)

```bash
curl -sL https://raw.githubusercontent.com/ovladon/wisdom-extractor/main/deploy/android/setup_phone.sh | bash
```

Installs python/git, the repo, the server-only dependencies (no Streamlit needed on the
phone), and the arm64 cloudflared. ~5–10 minutes on Wi-Fi.

## 4. Copy the database from the laptop (over home Wi-Fi, one time)

On the **laptop**:
```bash
cd ~/CSML/Conferinte/ConsIRL/ConsILR2025/wisdom-extractor/live
ip -4 addr | grep 192.168      # note the laptop's LAN IP
python3 -m http.server 8777
```
On the **phone** (Ubuntu prompt), with the laptop's IP:
```bash
curl -o /root/wisdom-live/wisdom.db http://192.168.X.Y:8777/wisdom.db
```
Then stop the laptop server with Ctrl+C (it was only reachable inside your Wi-Fi).

## 5. Start

```bash
bash /root/wisdom-extractor/deploy/android/start_phone.sh
```

Prints the public link + code. Also launches a **daily backup loop** (30 kept on the
phone) and the **weekly self-maintenance loop** (scrape 2 sources, glosses, years,
canonicalize, fold annotations into a fresh clustering).

## 6. Survive reboots (Termux:Boot)

In **Termux** (not Ubuntu — type `exit` first):
```bash
mkdir -p ~/.termux/boot
cat > ~/.termux/boot/start-wisdom <<'BOOT'
#!/data/data/com.termux/files/usr/bin/sh
termux-wake-lock
proot-distro login ubuntu -- bash /root/wisdom-extractor/deploy/android/start_phone.sh
BOOT
chmod +x ~/.termux/boot/start-wisdom
```
Now a power cut or reboot self-heals: phone boots → server + tunnel come back.

## 7. Optional but recommended: a permanent link via Netlify

The quick tunnel's URL changes on every restart. Fix: a tiny Netlify site that always
redirects to the current tunnel — the family bookmarks the Netlify link once, and the
phone updates the redirect automatically at every start.

1. In Netlify: **Add new site → Deploy manually** → drag in any text file → note the
   site's **API ID** (Site configuration → General) and its URL (you can rename it,
   e.g. `wisdom-lab.netlify.app`, or attach a subdomain of your own domain).
2. In Netlify: **User settings → Applications → New access token** → copy it.
3. On the phone (Ubuntu prompt) — you type your own token here, once:
```bash
cat > /root/.netlify_env <<'ENV'
NETLIFY_AUTH_TOKEN=paste-your-token-here
NETLIFY_SITE_ID=paste-the-api-id-here
ENV
chmod 600 /root/.netlify_env
```
From then on `start_phone.sh` refreshes the redirect automatically — the bookmark never
changes again.

## Updating to a new version

```bash
proot-distro login ubuntu
bash /root/wisdom-extractor/deploy/android/setup_phone.sh   # pulls + reinstalls deps
bash /root/wisdom-extractor/deploy/android/start_phone.sh
```

## Getting the database off the phone (for analysis on the laptop)

Reverse of step 4: on the phone `cd /root/wisdom-live && python3 -m http.server 8777`,
on the laptop `curl -o wisdom-from-phone.db http://PHONE_IP:8777/wisdom.db`, Ctrl+C on
the phone. Or use the daily backups in `/root/wisdom-live/backups/`.
