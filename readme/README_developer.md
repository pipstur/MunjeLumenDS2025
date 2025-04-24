## Standards of committing and branching on the repository
1. Developers create a feature branch from main.
```bash
git checkout -b name/feature
```
Example:
```bash
git checkout -b vojislav/initial-setup
```

2. Developers commit to this branch in the following manner:
```bash
git commit -m "action: short description"
```
Example:
```bash
git commit -m "add: initial infrastructure for the project"
```
The actions that should generally be used are: `add`, `update`, `fix`. Removing something constitutes updating it, so give an additional comment in that case.

3. When work is done (everything that's improtant for the feature is committed), they create a pull request.
```bash
git push origin name/feature
```
Example:
```bash
git push origin vojislav/initial-setup
```
4. CI/CD runs status checks (linting, tests, etc).
5. Code is reviewed and approved.
- This is important because the developers need to be up to date with what is being done on the project.
6. The PR is merged using rebase and fast-forward, keeping a linear history. When merging the pull request, you should select the option to `squash and merge`.
- The source branch can be deleted when merged.
7. No merge commits, ensuring a clean Git history.
8. When you are finished, to update the remote repo with your own:
```bash
git checkout main
git fetch --prune
git pull --rebase
```

General adding, committing tips:
- Use `git status` a lot to see what you're working with.
- Use `git tree` to see what branch you're checked out to (as well as the commit history), so there is no mixup to what branch it's being committed to.


The developer can continue to the `README_user.md` file for more information on environment setup, training, evaluation etc.
