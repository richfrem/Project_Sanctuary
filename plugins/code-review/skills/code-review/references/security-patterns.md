# Security Patterns Reference

Extracted from security guidance patterns. Check these when writing or reviewing code.

## Python Patterns

| Pattern | Risk | Safe Alternative |
|---------|------|-----------------|
| `eval()` | Code injection | `json.loads()`, `ast.literal_eval()` |
| `pickle.loads()` | Arbitrary code exec | `json.loads()` |
| `os.system(cmd)` | Shell injection | `subprocess.run([...], shell=False)` |
| `exec()` | Code injection | Avoid or sandbox |
| `__import__()` | Dynamic import abuse | Explicit imports |

## JavaScript/TypeScript Patterns

| Pattern | Risk | Safe Alternative |
|---------|------|-----------------|
| `eval()` | Code injection | `JSON.parse()` |
| `new Function()` | Code injection | Static functions |
| `innerHTML =` | XSS | `textContent` or sanitize with DOMPurify |
| `document.write()` | XSS | `createElement()` + `appendChild()` |
| `dangerouslySetInnerHTML` | XSS | Sanitize with DOMPurify |
| `child_process.exec()` | Shell injection | `execFile()` with array args |

## GitHub Actions Patterns

| Pattern | Risk | Safe Alternative |
|---------|------|-----------------|
| `${{ github.event.issue.title }}` in `run:` | Command injection | Use `env:` block |
| `${{ github.event.pull_request.body }}` in `run:` | Command injection | Use `env:` block |

**Unsafe:**
```yaml
run: echo "${{ github.event.issue.title }}"
```

**Safe:**
```yaml
env:
  TITLE: ${{ github.event.issue.title }}
run: echo "$TITLE"
```

## Risky GitHub Actions Inputs
- `github.event.issue.body`
- `github.event.pull_request.title`
- `github.event.pull_request.body`
- `github.event.comment.body`
- `github.event.head_commit.message`
- `github.head_ref`

## General Rules
- Never trust user-controlled input in shell commands
- Always use parameterized queries for databases
- Validate and sanitize all external input
- Use allowlists over denylists
- Log security events, never log secrets
