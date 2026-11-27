# Contributing

This is a two-person academic project. These rules keep the history clean and every
result reproducible.

## Workflow

1. **Sync before you start**

   ```bash
   git checkout main && git pull
   dvc pull
   ```

2. **One branch per task**, named `<type>/<short-description>`:
   `feat/efficientnet`, `fix/split-leak`, `docs/model-card`, `exp/augmentation-ablation`.

3. **Commit small, meaningful changes** using [Conventional Commits](https://www.conventionalcommits.org/):

   ```
   feat(models): add DenseNet-121 builder
   fix(data): clip intensities before min-max scaling
   docs: describe the patient-level split
   ```

4. **Before pushing**

   ```bash
   make lint test          # style + unit tests
   dvc repro               # if you changed the data pipeline
   dvc push                # ALWAYS before git push when data changed
   git push -u origin <branch>
   ```

5. **Open a pull request** to `main`. CI must pass and the other teammate reviews it.

## Rules

- **Logic lives in `src/brain_tumor/`.** Notebooks only call it; they are hard to review and merge.
- **No hard-coded hyperparameters.** Use `params.yaml` or command-line overrides.
- **Never commit data, weights or secrets.** Data goes through DVC, weights through MLflow,
  tokens stay in `.env` / `.dvc/config.local` (both git-ignored).
- **The test set is evaluated once per final run.** Model selection and tuning use the
  validation set only.
- **MLflow naming:** one experiment per model (`brain-tumor/<model>`), runs named
  `<model>-<variant>-<initials>`, e.g. `resnet18-aug-ys`.

## Adding a model

1. Implement a builder in `src/brain_tumor/models/` returning an `nn.Module` with 3 outputs.
2. Register it in `models/factory.py`.
3. Add a shape test in `tests/test_models.py`.
4. Document it in `docs/models.md`.
