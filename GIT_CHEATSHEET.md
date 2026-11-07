# Git Cheat Sheet for the Team

Think of Git as "save points" for your code, and GitHub as the shared copy online.

- **commit** = save a snapshot on your computer
- **push** = upload your commits to GitHub
- **pull** = download your teammates' commits from GitHub

## One-time setup (each person)

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"   # same email as your GitHub account
git clone <repo-url>
cd hackutd-project
```

## The daily loop (do this over and over)

```bash
git pull                       # 1. get the latest from teammates, ALWAYS FIRST
# ... edit code ...
git status                     # 2. see what you changed
git add .                      # 3. stage everything you changed
git commit -m "Add login page" # 4. save a snapshot with a short message
git pull                       # 5. grab anything new again
git push                       # 6. upload to GitHub
```

Commit small and often (every 20-30 minutes), and push at least every hour.

## Avoiding conflicts (the #1 hackathon problem)

- **Split up files.** Frontend person edits frontend files, backend person edits backend files. Two people editing the same file at the same time is how conflicts happen.
- **Pull before you start working** and before you push.
- **Tell the team in chat** before you touch a shared file.

## Branches (optional, but safer)

If you'd like to avoid breaking `main`, work on your own branch:

```bash
git checkout -b my-feature     # create and switch to a new branch
git push -u origin my-feature  # first push of that branch
```

Then on github.com, click "Compare & pull request" and then "Merge" when it works.
For a 24-hour hackathon, committing straight to `main` is fine if you split up files.

## If something goes wrong

| Problem | Fix |
|---|---|
| `git push` is rejected ("fetch first") | Run `git pull`, then `git push` again |
| Merge conflict (`<<<<<<<` markers in a file) | Open the file, keep the code you want, delete the marker lines, then `git add .` and `git commit -m "Fix conflict"` |
| I want to undo my edits to a file (not committed yet) | `git restore path/to/file` (this throws away your edits) |
| I committed a secret / API key | Tell the team immediately and **revoke that key** on the provider's site. Deleting the file isn't enough |
| I'm totally lost | `git status` tells you what's going on. Ask a teammate or Claude before running anything with `--force` or `reset --hard` |

## Never do

- `git push --force` (can erase teammates' work)
- Commit `.env` or any API key
- Commit `node_modules/` (it's already in `.gitignore`)
