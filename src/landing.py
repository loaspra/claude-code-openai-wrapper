def render_landing_page(version: str, auth_method: str, auth_valid: bool) -> str:
    status_text = "Connected" if auth_valid else "Not Connected"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Claude Code OpenAI Proxy</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; max-width: 840px; }}
    code, pre {{ background: #f4f4f5; padding: .15rem .3rem; border-radius: .25rem; }}
    pre {{ padding: 1rem; overflow-x: auto; }}
    .status {{ padding: .75rem 1rem; border: 1px solid #d4d4d8; border-radius: .5rem; }}
  </style>
</head>
<body>
  <h1>Claude Code OpenAI Proxy</h1>
  <p>Small OpenAI/Anthropic-compatible proxy backed by Claude Code credentials.</p>
  <p class="status"><strong>{status_text}</strong> via <code>{auth_method}</code>, version <code>{version}</code></p>
  <h2>Core Endpoints</h2>
  <ul>
    <li><code>POST /v1/chat/completions</code></li>
    <li><code>POST /v1/messages</code></li>
    <li><code>GET /v1/models</code></li>
    <li><code>GET /v1/auth/status</code></li>
    <li><code>GET /health</code></li>
  </ul>
  <h2>Quick Test</h2>
  <pre>curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{{"model":"claude-sonnet-4-6","messages":[{{"role":"user","content":"Hello"}}]}}'</pre>
  <p><a href="/docs">OpenAPI docs</a></p>
</body>
</html>"""
