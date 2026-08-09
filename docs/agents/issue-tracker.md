# Issue tracker: GitHub

Issues and specifications for this repository live in GitHub Issues at
`jasmineee-li/warp`. Use the `gh` CLI for operations.

## Conventions

- Create: `gh issue create --title "..." --body "..."`
- Read: `gh issue view <number> --comments`
- List: `gh issue list --state open`
- Comment: `gh issue comment <number> --body "..."`
- Add a label: `gh issue edit <number> --add-label "..."`
- Remove a label: `gh issue edit <number> --remove-label "..."`
- Close: `gh issue close <number> --comment "..."`

Run commands from this repository so `gh` resolves the remote automatically.

## Pull requests as a request surface

PRs as a request surface: no.

Pull requests are implementation and review surfaces. They do not enter the
issue-triage queue unless this setting is changed later.

## Skill operations

When a skill says “publish to the issue tracker,” create a GitHub issue.

When a skill says “fetch the relevant ticket,” read the GitHub issue and its
comments and labels.
