# Publishing `spectraltm-db` to PyPI (production)

> **Read this end-to-end before publishing.** Once a version is on
> PyPI, it is permanently visible and **cannot be re-uploaded under
> the same version**. All mistakes become immutable.

This document covers the production flow for pushing `spectraltm-db`
to https://pypi.org/. It does **not** use TestPyPI — TestPyPI is a
separate endpoint meant for CI smoke testing; if you want TestPyPI
practice first, swap `--repository pypi` for `--repository testpypi`
in step 6 (and create a separate API token + URL in `~/.pypirc`).

---

## 0. Pre-flight checklist (do all of these before touching PyPI)

```text
[ ]  All tests green locally:
       pytest tests/ -q

[ ]  Wheel builds clean:
       python -m pip install --upgrade build twine
       rm -rf dist/ build/ spectraltm_db.egg-info/
       python -m build

       Expect:
         dist/spectraltm_db-X.Y.Z-py3-none-any.whl
         dist/spectraltm_db-X.Y.Z.tar.gz

[ ]  twine is happy with the metadata:
       python -m twine check dist/*

       Expect: "PASSED" for both files. Any "FAIL" with a missing
       description / long_description / classifer note = bug.

[ ]  CHANGELOG.md has the release date and final entry:
       ## [X.Y.Z] — YYYY-MM-DD
       ...

[ ]  README.md final (badges, install instructions, citation, license).

[ ]  Git working tree is clean + tagged:
       git tag -a vX.Y.Z -m "release X.Y.Z"
       git push --tags

[ ]  Version bumped in pyproject.toml:
       [project]
       version = "X.Y.Z"
       (no .dev0 / .rc1 suffix on what you push to PyPI)

[ ]  A PyPI API token with project-scope "Upload packages" for
       `spectraltm-db` exists in https://pypi.org/manage/account/token/.
```

---

## 1. Get a PyPI API token (once)

This is a one-time setup. The token replaces your PyPI password;
publish from a long-lived local machine using an **account-wide token**
if you also want to publish other projects from it, or a **project-scoped
token** if you only publish `spectraltm-db`. Project-scoped is safer.

1. Open https://pypi.org/manage/account/token/.
2. **Add API token** → scope: **Project: spectraltm-db** (or
   Account-wide if no project scope exists yet — `spectraltm-db` must
   be registered as a project on your account first; PyPI creates the
   placeholder the moment you upload a file with the matching name).
3. PyPI shows the token **once**. Copy it before leaving the page;
   it looks like `pypi-XXXXXXXXXXXXXXXX` and is unrecoverable later.

---

## 2. Configure twine

**Primary — `$env:USERPROFILE\.pypirc` (Windows) / `~/.pypirc`
(POSIX)**. This is where every long-lived local install of twine reads
its credentials. The token stays in this one file and every `twine
upload` reads from it automatically.

Create or extend the file:

```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-XXXXXXXXXXXXXXXX
```

Open it in Notepad / your editor of choice:

```powershell
notepad $env:USERPROFILE\.pypirc
# or
code $env:USERPROFILE\.pypirc
```

Replace `pypi-XXXXXXXXXXXXXXXX` with the actual token from step 1.
The literal string `__token__` (with the double underscores) is
mandatory — that's PyPI's convention for token-based auth.

Twine reads this file automatically; you don't need to invoke any
env vars.

**Fallback — environment variables** (CI only). On GitHub Actions,
GitLab CI, or any non-interactive system that should not carry a
`~/.pypirc` file, expose:

```bash
TWINE_USERNAME = __token__
TWINE_PASSWORD = pypi-XXXXXXXXXXXXXXXX
```

`twine` reads these variables directly when no `.pypirc` is
present. Don't use this locally — point 1 of this guide is "keep
the token out of chat history / shell history / env dumps", which
env vars don't satisfy on a shared dev machine.

**POSIX file mode:** on Linux / macOS / Git Bash, lock the file
down so other users on the box can't read it:

```bash
chmod 600 ~/.pypirc
```

On Windows the file mode is governed by NTFS ACLs instead. The
default ACL PyPI checks is `users only`; if you share the machine
with other Windows accounts, right-click → Properties → Security →
remove the `Users` group from the access list and add only your own
user.

**Verifying twine will read the file:** `twine upload` will *not*
echo your token, but you can confirm the file is readable with:

```bash
python -c "import configparser; c=configparser.ConfigParser(); \
  c.read('$HOME/.pypirc' if not 'USERNAME' in __import__('os').environ \
       else r'%USERPROFILE%\.pypirc'); print(c['pypi']['username'])"
# → __token__
```

If the username prints as `__token__`, twine will see the credentials.

---

## 3. Bump version + write the release entry

Edit `pyproject.toml`:

```toml
[project]
name = "spectraltm-db"
version = "X.Y.Z"            # ← change this
```

Add a `## [X.Y.Z] — YYYY-MM-DD` block at the top of `CHANGELOG.md` with
the released features and any breaking changes called out under
`### Changed` (per Keep-a-Changelog semantics).

Verify no `__pycache__`, `.pyc`, or scratch files made it into the
package source — `pyproject.toml`'s `[tool.setuptools.packages.find]`
includes `spectraltm_db*`, which catches the package directory but not
random artifacts at the repo root.

```bash
# Confirm the package source is clean
git status
```

---

## 4. Build the distribution

```bash
python -m pip install --upgrade build
rm -rf dist/ build/ spectraltm_db.egg-info/
python -m build
```

Output:

```
dist/
├── spectraltm_db-X.Y.Z-py3-none-any.whl
└── spectraltm_db-X.Y.Z.tar.gz
```

Both are required by PyPI's source distribution policy. Confirm:

```bash
python -m twine check dist/*
# → Checking distribution dist/spectraltm_db-X.Y.Z-py3-none-any.whl: PASSED
# → Checking distribution dist/spectraltm_db-X.Y.Z.tar.gz:        PASSED
```

A failure here means `pyproject.toml` is missing required fields. The
most common offender is forgetting `version`, `description`, or
`readme`. The README presence is verified by `twine check` and is
strictly enforced for `setuptools>=68`.

---

## 5. Verify before upload (dry run)

Inspect the wheel's metadata to make sure PyPI will accept it:

```bash
unzip -p dist/spectraltm_db-X.Y.Z-py3-none-any.whl METADATA | head -50
```

Look for:

- `Version: X.Y.Z` (matches pyproject.toml)
- `Name: spectraltm-db` (lowercase + dash; matches PyPI slot)
- `Requires-Python: >=3.10`
- `Requires-Dist: spectraltm>=0.1.1`
- `License-File: LICENSE`

If anything looks off, fix the `pyproject.toml` and rebuild.

Smoke-test the install in a fresh venv so the published metadata
matches reality:

```bash
python -m venv /tmp/publish-smoke
/tmp/publish-smoke/bin/pip install dist/spectraltm_db-X.Y.Z-py3-none-any.whl
/tmp/publish-smoke/bin/python -c "import spectraltm_db; print(spectraltm_db.__version__)"
# → X.Y.Z
/tmp/publish-smoke/bin/python -c "
import spectraltm_db as stm, random
random.seed(0)
cal = [random.gauss(0,1) for _ in range(200*32)]
idx = stm.Index.create(name='x', path='/tmp/smoke', dimension=32,
                       compression='spectral_k64', calibration_sample=cal)
idx.upsert(vectors=[{'id': 'a', 'values': [random.gauss(0,1) for _ in range(32)]}])
r = idx.query(vector=[0.1]*32, top_k=1)
print(r['matches'])
"
```

---

## 6. Upload to PyPI — production

If section 2 is set up (`.pypirc` exists at `$env:USERPROFILE\.pypirc`
with the `[pypi]` block), twine reads credentials automatically:

```bash
python -m twine upload dist/*
# or, explicit and safer:
python -m twine upload dist/spectraltm_db-X.Y.Z-py3-none-any.whl \
                     dist/spectraltm_db-X.Y.Z.tar.gz \
                     --repository pypi
# When --repository is omitted twine defaults to https://upload.pypi.org/legacy/
# which is the *production* endpoint. There's no implicit TestPyPI here.
```

A successful upload looks like:

```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading spectraltm_db-X.Y.Z-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18/18 kB
Uploading spectraltm_db-X.Y.Z.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14/14 kB

View at: https://pypi.org/project/spectraltm-db/X.Y.Z/
```

If `.pypirc` is missing or malformed, twine prompts for
`username` (`__token__`) and `password` (the API token from step 1)
on the first upload — confirm without typing anything else and the
upload still works. The fix is to populate `.pypirc` afterwards
so subsequent uploads are silent.

A successful upload looks like:

```
Uploading distributions to https://upload.pypi.org/legacy/
Uploading spectraltm_db-X.Y.Z-py3-none-any.whl
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18/18 kB
Uploading spectraltm_db-X.Y.Z.tar.gz
100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 14/14 kB

View at: https://pypi.org/project/spectraltm-db/X.Y.Z/
```

---

## 7. Verify the published artifact

PyPI propagation is fast but not instant — give it 30-60 seconds.

```bash
# Index sees the new version:
pip index versions spectraltm-db

# Install from the real PyPI (not the local wheel) into a fresh venv:
python -m venv /tmp/published-smoke
/tmp/published-smoke/bin/pip install --upgrade spectraltm-db
/tmp/published-smoke/bin/python -c "import spectraltm_db; print(spectraltm_db.__version__)"
```

Open https://pypi.org/project/spectraltm-db/ in a browser and verify:

- The README renders correctly (PyPI serves `README.md` as long-description).
- Badges and the citation block are present.
- The "Download files" tab shows both `*.whl` and `*.tar.gz`.
- The "Project Links" sidebar shows the metadata you set in `pyproject.toml`.

---

## 8. Tag the release

The git tag is what links the version on PyPI back to source. Do this
*after* PyPI accepts the upload.

```bash
git tag -a vX.Y.Z -m "release X.Y.Z — short note"
git push origin main
git push origin vX.Y.Z
```

On GitHub, draft a release from `vX.Y.Z` with the CHANGELOG entry as
notes. The wheel/tar.gz that PyPI shows are downloadable directly, but
GitHub Releases is what readers tend to find first.

---

## 9. Common failure modes

| Symptom                                              | Cause                                                       |
|-------------------------------------------------------|-------------------------------------------------------------|
| `403 Forbidden: You are not allowed to edit 'spectraltm-db'` | API token scope doesn't include this project, or it belongs to a different PyPI account. |
| `400 File already exists`                              | You're trying to re-upload the same filename. Bump the version in `pyproject.toml`. |
| `400 The description failed to render`                 | The README contains reStructuredText that breaks. PyPI supports Markdown via `long_description_content_type = "text/markdown"` — confirm that's set in `pyproject.toml`. |
| `twine check` reports a missing `Project-URL`          | Add `[project.urls] Homepage = "..."` (etc.) under `pyproject.toml`. |
| The published wheel installs but `import spectraltm_db` raises `ModuleNotFoundError` | The package code wasn't included. Open the wheel with `unzip -l dist/*.whl` — if it contains only `dist-info/` and no top-level `spectraltm_db/` directory, your `[tool.setuptools.packages.find]` config is broken. The fix is to remove `where=…` and any explicit `package-dir = { "" = … }` — the default discovery works correctly for this layout. |
| `wheel install ok, but import raises an obscure site-packages error` | Check the wheel was actually rebuilt after the `pyproject.toml` fix: `rm -rf dist/ build/ *.egg-info && python -m build` is non-negotiable before re-testing. |

---

## 10. After publishing

You cannot delete a version from PyPI, but you *can* yank it (hide from
`pip install` while keeping the file downloadable for users who pinned
it). Use Yanking for broken releases; never re-upload a different file
under the same version.

To yank:

```bash
# Web only as of writing — open https://pypi.org/project/spectraltm-db/
# → "Manage project" → "Releases" → check the version → "Yank".
```

Ship a fix as `X.Y.Z+1` with the original CHANGELOG entry marked as
yanked. Existing users pinned to `X.Y.Z` continue to work; new users
land on the fixed version.
