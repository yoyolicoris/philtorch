---
description: >-
  Review a philtorch pull request or diff with the philtorch-code-review skill.
  Reads files and runs read-only git and gh; never edits, approves or merges.
mode: all
model: model_api/muse-spark-1.3-contributor
tools:
  read: true
  grep: true
  glob: true
  list: true
  bash: true
  write: false
  edit: false
  patch: false
  webfetch: false
  task: false
permission:
  edit: deny
  webfetch: deny
  external_directory: deny
  # The last matching rule wins: the catch-all comes first.
  bash:
    "*": deny
    "git diff*": allow
    "git show*": allow
    "git log*": allow
    "git blame*": allow
    "git status*": allow
    "gh pr view*": allow
    "gh pr diff*": allow
    "git * --output*": deny
---
Review the pull request or diff you are given by following
`.github/skills/philtorch-code-review/SKILL.md`.
