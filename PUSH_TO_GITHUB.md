# Publishing the repository (one-time, ~5 minutes)

1. On github.com create a new **public** repository named `truminds-traffic` (no README, no .gitignore).
2. Open **PowerShell** in `D:\Downloads\truminds-traffic` and run:

```powershell
git init
git lfs install 2>$null   # optional; not required, all files are < 100 MB
git add .
git commit -m "TruMinds: traffic event detection and accident anticipation"
git branch -M main
git remote add origin https://github.com/trumen-com/road-vision.git
git push -u origin main
```

3. Tell Claude the repository URL. It will then:
   - create the Vercel project for the website (root directory `web/`),
   - create the Railway service for the live-demo backend (`demo/Dockerfile`),
   - fill the links on the website and in the README.

Before the deadline (Sunday 23:59 Tashkent), tag the final commit:

```powershell
git tag submission-v1
git push origin submission-v1
```
