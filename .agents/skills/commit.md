# Skill: Commit standard for ASL-System

## Format

```text
type(scope): short description in English
```

Use Conventional Commits style.

## Allowed types

| Type | Use |
|---|---|
| `feat` | New functionality |
| `fix` | Bug fix |
| `refactor` | Internal restructuring without intended behavior change |
| `docs` | Documentation only |
| `test` | Tests added or corrected |
| `chore` | Tooling, scripts, config, maintenance |
| `style` | Formatting or visual-only changes without logic changes |
| `perf` | Performance improvement |
| `build` | Build pipeline or dependency changes |
| `ci` | CI workflow changes |

## Suggested scopes

| Scope | Use |
|---|---|
| `web` | `ASL-Web` frontend |
| `web-server` | `ASL-Web/server` |
| `mobile` | `ASL-MobileAPP` |
| `callapp` | `ASL-CallAPP/app` |
| `call-server` | `ASL-CallAPP/server` |
| `ia` | `ASL-IA` |
| `agents` | `.agents/` |
| `infra` | root scripts, Docker, git, workspace |

## Rules

1. Write the commit message in English.
2. Keep one logical change per commit.
3. Describe what changed, not how you changed it.
4. Do not end the subject with a period.
5. Use the dominant scope when one module clearly owns the change.
6. Use `infra` for cross-cutting repo-level work.

## Examples

```text
feat(web): add request follow-up flow
fix(mobile): correct websocket petition submission
refactor(call-server): simplify interpreter presence handling
docs(agents): align templates with the ASL monorepo
chore(infra): update local startup script paths
```
