# MLOps setup — step by step

One-time setup for the team, then the daily workflow. Replace `<user>` with your
GitHub/DagsHub username.

## 1. Local environment

Ubuntu prerequisites (once): `sudo apt install python3-venv git make`.

```bash
cd ~/Documents/S1/Deep_Learning/projet/brain-tumor-mlops
bash scripts/bootstrap.sh
source .venv/bin/activate
```

The script creates `.venv`, installs the CPU build of PyTorch when no NVIDIA GPU is
present (training happens on Colab), installs the project and dev tools, runs
`git init` and `dvc init`, copies `.env.example` to `.env`, installs the pre-commit
hooks, runs the tests and makes the first commit. It is safe to re-run.

## 2. Git and GitHub

The bootstrap script already created the repository and the first commit:

```bash
git remote add origin https://github.com/<user>/brain-tumor-mlops.git
git push -u origin main
```

On GitHub:

- *Settings → Branches → Add rule* for `main`: require a pull request and the **CI** checks.
- Invite your teammate as a collaborator.

## 3. DagsHub (DVC storage + MLflow server)

1. Sign in to [dagshub.com](https://dagshub.com) with GitHub.
2. *Create → New Repository → Connect a repository → GitHub* and pick `brain-tumor-mlops`.
   DagsHub mirrors the code automatically.
3. *User settings → Tokens*: create a token. Put it in `.env` (copy of `.env.example`).
4. Note the two URLs shown under the repository's **Remote** button:
   - DVC: `https://dagshub.com/<user>/brain-tumor-mlops.dvc`
   - MLflow: `https://dagshub.com/<user>/brain-tumor-mlops.mlflow`

## 4. DVC

```bash
dvc remote add -d origin https://dagshub.com/<user>/brain-tumor-mlops.dvc
# credentials stay local (.dvc/config.local is git-ignored)
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <dagshub-token>
git add .dvc/config .dvcignore
git commit -m "chore: configure DVC remote"

dvc repro                  # builds the dataset (~1 GB download, a few minutes)
git add dvc.lock
git commit -m "data: first version of the dataset"
dvc push                   # upload data to DagsHub
git push
```

## 5. MLflow

Fill in `.env`:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/brain-tumor-mlops.mlflow
MLFLOW_TRACKING_USERNAME=<user>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>
```

`brain_tumor.tracking.setup_mlflow` loads `.env` automatically. Without these variables,
runs go to a local SQLite file (`mlflow.db`); view them with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## 6. Google Colab

1. Open `notebooks/02_colab_runner.ipynb` from GitHub in Colab
   (*File → Open notebook → GitHub*).
2. Add secrets (key icon): `DAGSHUB_USER`, `DAGSHUB_TOKEN`, optionally `GITHUB_TOKEN`
   (a fine-grained token with *contents: write* on this repository only).
3. *Runtime → Change runtime type → GPU*, then run the cells.

## 7. GitHub Actions

`.github/workflows/ci.yml` runs on every push and pull request:

| Job | What it does |
|---|---|
| `lint` | Ruff lint + format check |
| `test` | CPU PyTorch + full pytest suite on synthetic data (includes a tiny training run, Optuna and the API) |
| `validate-pipeline` | `dvc stage list` — checks that `dvc.yaml` is valid |

No secret is needed: tests never touch the real data or the remote MLflow server.

## Daily workflow

```bash
git checkout main && git pull && dvc pull          # sync
git checkout -b feat/<task>                        # one branch per task
# ... edit code ...
make lint test
dvc repro                                          # if the data pipeline changed
git add -A && git commit -m "feat: ..."
dvc push                                           # BEFORE git push if data changed
git push -u origin feat/<task>                     # open a pull request
```

Training happens on Colab and logs straight to MLflow, so nothing needs to be copied back,
except tuned parameters (`configs/best_params/`), which are committed.

## Troubleshooting

| Problem | Fix |
|---|---|
| `dvc pull` asks for a password / 401 | Re-run the three `dvc remote modify origin --local …` commands |
| MLflow 401 | Check `MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` in `.env` or Colab |
| `FileNotFoundError: data/processed/...` | `dvc pull` (or `dvc repro preprocess`) |
| `ConfigError: Unknown config key` | Typo in a command-line override; check `params.yaml` |
| CUDA out of memory | `train.batch_size=16` |
| Teammate's `dvc pull` misses files | The author forgot `dvc push` before `git push` |
