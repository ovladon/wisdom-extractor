# Running the Wisdom Extractor on Windows (for annotators)

## The 3-step version

1. Download the app: green **Code** button on the GitHub page → **Download ZIP** →
   right-click the ZIP → **Extract All**.
2. Open the extracted folder and **double-click `run_windows.bat`**.
   - First time only: it installs what it needs (~5 minutes; if Python is missing it
     installs that too, then asks you to double-click once more).
   - A black window appears — that's the app running. **Leave it open.**
3. Your browser opens the app by itself. Go to the **"6) Annotate • Play"** tab,
   type your name in the left sidebar (that's your leaderboard identity), and judge
   pairs: ✅ same idea · 🚫 different idea · ❌ not a saying · ⏭️ skip.

To stop: close the black window. To start again: double-click `run_windows.bat`.

## If SmartScreen complains
Windows may warn about an unrecognised app the first time. Click **More info → Run anyway**
(the launcher is a plain, readable script — open it in Notepad if you're curious).

## For the study coordinator
- Each annotator's judgments are saved in the local `wisdom.db`. Collect them via
  tab **9) Export → annotation_export.json** and merge centrally, or — much better —
  host one shared instance so everyone annotates the same database (see README;
  a small VPS or a tunnel like Tailscale Funnel pointed at one running instance).
- Seed the database before annotators start: tab 2 → "Seed from paper dataset".
