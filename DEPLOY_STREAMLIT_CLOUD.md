# Deploy to Streamlit Community Cloud (share a link with anyone)

This guide puts your app online for **free** so your friend just clicks a
link in their browser — no install, no Terminal, works on Mac, Windows,
phone, anything.

The app has already been made **deploy-ready** in the code, so you only
need to follow the clicking steps below. Total time: about 15–20 minutes
the first time.

---

## What you'll need (all free)

1. A **GitHub account** — sign up at https://github.com/signup
2. **GitHub Desktop** app (the easy, no-Terminal way to upload code) —
   download at https://desktop.github.com
3. Your **Gemini API key** and **admin password** (already in your `.env`)

---

## Step 1 — Put the project on GitHub

> ⚠️ **Never upload your `.env` file.** It holds your real API key and
> password. Good news: the project's `.gitignore` already blocks `.env`,
> `venv/`, and `vector_store/` automatically, so if you use GitHub Desktop
> (or git) they will be skipped for you. Just don't drag files in by hand.

1. Install and open **GitHub Desktop**, then sign in with your GitHub account.
2. Click **File → Add local repository**, and choose the folder
   `healthcare-knowledge-assistant`.
3. It will say the folder isn't a git repository yet — click
   **"create a repository"** (a link in that message).
4. In the dialog, keep the name, and click **Create repository**.
5. You'll see a list of files to include. Confirm that **`.env` is NOT in
   the list** (it should be hidden automatically). If you ever see `.env`
   listed, stop and uncheck it.
6. At the bottom left, type a summary like `Initial commit` and click
   **Commit to main**.
7. Click **Publish repository** at the top.
   - You can leave **"Keep this code private"** either way — Streamlit can
     read private repos. Private is safer.
   - Click **Publish repository**.

Your code is now on GitHub. 🎉

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io and click **Sign in** → **Continue
   with GitHub**. Approve the access request.
2. Click **Create app** (or **New app**) → **Deploy a public app from a
   repository** (works for private repos too when signed in with GitHub).
3. Fill in:
   - **Repository:** `your-username/healthcare-knowledge-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Advanced settings** and set **Python version** to **3.11**.
5. **Don't click Deploy yet** — first add your secrets in the same
   Advanced settings box (next step).

---

## Step 3 — Add your secrets (this replaces the `.env` file)

In the **Advanced settings → Secrets** box, paste the following, replacing
the placeholder values with your **real** ones (copy them from your `.env`):

```toml
GEMINI_API_KEY = "paste-your-real-key-here"
GEMINI_MODEL = "gemini-2.0-flash"
ADMIN_PASSWORD = "paste-your-real-admin-password-here"
```

> 💡 **Why `gemini-2.0-flash` and not `gemini-flash-latest`?** The "latest"
> alias currently points to a very busy model that keeps returning
> "high demand, try again later" errors and has a tiny free daily limit
> (about 20 questions). `gemini-2.0-flash` is far more available and has a
> much larger free quota — a better choice for a shared demo. You can use
> it locally too by changing the same line in your `.env`.

Now click **Save**, then **Deploy**.

The first launch takes a few minutes: it installs the libraries, then
builds the search index automatically (you'll briefly see
"Setting up the assistant for first use..."). When it's done, you get a
public URL like:

```
https://your-app-name.streamlit.app
```

**That link is what you send your friend.**

---

## Step 4 (optional) — Control who can open it

By default anyone with the link can use the app (and each question uses
your Gemini quota). To restrict it:

1. In the Streamlit dashboard, open your app → **Settings → Sharing**.
2. Turn on **"Only specific people can view this app"**.
3. Add your friend's Google email. Now only invited people can open it.

Either way, the **admin panel stays password-protected** with your admin
password, so no visitor can rebuild the index or see system status.

---

## After it's live

- **To send it:** just share the `https://...streamlit.app` link.
- **If you edit the knowledge base later:** change the `.md` files, commit
  in GitHub Desktop, and click **Push origin**. Streamlit auto-redeploys,
  and the app rebuilds its index on the next boot.
- **To change the key or password:** edit them in the app's
  **Settings → Secrets** box (not in the code), then reboot the app from
  the dashboard.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "high demand" / 503 errors | Google's model is busy. Use `GEMINI_MODEL = "gemini-2.0-flash"` in secrets (Step 3). |
| "quota exceeded" / 429 | You hit the free daily limit. Wait 24h, or use `gemini-2.0-flash` for a bigger allowance. |
| App shows "unavailable" to visitors | Check the **Secrets** box has the API key spelled exactly `GEMINI_API_KEY`. Reboot the app. |
| Build fails with a memory error | The ML libraries are heavy. If Streamlit Cloud runs out of memory, **Hugging Face Spaces** (free, more memory) is a good alternative — ask and I'll write that guide. |
| You accidentally committed `.env` | Delete the repo, **regenerate your API key** at https://aistudio.google.com/apikey, and start Step 1 again. |

---

## Quick security recap

- ✅ Secrets live in Streamlit's **Secrets** box, never in the code or GitHub.
- ✅ `.gitignore` keeps `.env` off GitHub automatically.
- ✅ The admin panel is password-protected even on the public app.
- ✅ The knowledge base is fictional sample data — no real patient info.
