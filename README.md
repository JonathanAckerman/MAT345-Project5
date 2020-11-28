# MAT345-Project5
Neural net for handwritten digit recognition

## Notes
Dataset is not included in the repo

## Workflow
### 1) Get latest changes from the repo
```
git fetch           // See if there are any remote changes
git pull --ff-only  // Fast forward to latest if possible (you should have no local changes)
```

### 2) Do some work

### 3) Make some commits
```
git status                                            // See what has been changed locally
git add _<filename>_                                  // Track changed file (replace <filename> with the -A flag to track all changed files)
git commit -m "Commit message goes here in quotes"    // Commit changes
```

### 4) Integration
```
git pull --rebase   // Rebase any changes (This applies remote commits first, then your commits on top)
    // There shouldn't be any issues but if there are this is where they should be caught
git push            // Push to remote
```
