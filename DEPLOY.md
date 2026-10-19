# Deploying CineMatch (free, with your key hidden)

Result: a public link like `https://cinematch-yourname.streamlit.app`. Visitors never see or
enter an API key — it lives only in the host's private "Secrets" setting.

**Before you start:** the app must already run correctly on your laptop
(`python -m streamlit run app.py`).

---

## 1. Check that no secret is in your folder
Your TMDB key must not appear in any file you upload. Search for it:

```powershell
Select-String -Path *.py,*.md,*.txt -Pattern "your_actual_key_here"
```
It should print nothing. (`.env` and `.streamlit\secrets.toml` are already in `.gitignore`.)

## 2. Put the project on GitHub
1. Create a free account at github.com and install Git from git-scm.com.
2. On github.com click **New repository** → name it `cinematch` → **Public** → *don't* tick
   "Add a README" → **Create repository**.
3. In PowerShell, inside your project folder:
```powershell
git init
git add .
git commit -m "CineMatch"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cinematch.git
git push -u origin main
```
4. Check what got uploaded:
```powershell
git ls-files
```
The list must **not** include `.env` or `.streamlit/secrets.toml`. (`secrets.toml.example` is fine.)

> If Git asks for a name/email first, run:
> `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`

## 3. Deploy on Streamlit Community Cloud
1. Go to **share.streamlit.io** and sign in with GitHub.
2. Click **Create app** (deploy from an existing repo) and choose:
   - Repository: `YOUR_USERNAME/cinematch`
   - Branch: `main`
   - Main file path: `app.py`
3. Open **Advanced settings**:
   - **Python version:** pick **3.12**
   - **Secrets:** paste exactly this line, using your real key:
     ```
     TMDB_API_KEY = "your_real_tmdb_key"
     ```
4. Click **Deploy**. The first build takes a few minutes (installing libraries).

*(Button names can shift slightly as Streamlit updates its site; the settings above are what matters.)*

## 4. After it's live
- Open the link in a private/incognito window: you should see **no key field in the sidebar**.
- Try a movie. The **first search is slow** (about 10–20 s): it builds the live movie pool,
  which is then cached for an hour.
- Apps with no visitors go to sleep; the next visitor sees a "wake up" button and waits ~30 s.
- Change the key later in **App settings → Secrets** (no code change).

## What's different from your laptop version
| | Laptop | Public site |
|---|---|---|
| API key | typed into sidebar | hidden, from Secrets |
| Matching | semantic + keyword (if installed) | **keyword (TF-IDF) only** |
| Pool size slider | available | fixed (protects the shared key) |

Semantic matching is excluded online because PyTorch is too heavy for the free tier — demo
that mode from your laptop and show screenshots of both.

## Limits to know
- All visitors share one TMDB key (about 40 requests / 10 s). The app throttles and caches, but a
  burst of many simultaneous visitors can slow responses.
- TMDB's terms are for non-commercial use. Keep the attribution line at the bottom of the app.
- If the site shows a "TMDB rejected the API key" message, re-check the Secrets value
  (no extra spaces, keep the quotes).
