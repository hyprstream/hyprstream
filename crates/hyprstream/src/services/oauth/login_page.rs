//! Provider selection login page for the OAuth authorize flow.
//!
//! Renders an HTML page listing available authentication methods:
//! external OIDC providers and local Ed25519 challenge-response.

/// Render the login page with provider selection.
///
/// `providers` is a list of `(slug, display_name)` pairs for external OIDC providers.
/// `authorize_query` is the original authorize query string to pass through.
pub fn render_login_page(
    client_name: &str,
    scopes: &str,
    redirect_host: &str,
    authorize_query: &str,
    providers: &[(String, String)],
    has_local_auth: bool,
) -> String {
    let mut provider_buttons = String::new();
    for (slug, display_name) in providers {
        provider_buttons.push_str(&format!(
            r#"<a class="provider-btn" href="/oauth/external/authorize/{slug}?{query}">
                Sign in with {display_name}
            </a>"#,
            slug = html_escape(slug),
            query = authorize_query,
            display_name = html_escape(display_name),
        ));
    }

    let local_section = if has_local_auth {
        let collapsed = if providers.is_empty() { "" } else { "collapsed" };
        format!(
            r#"<details class="local-auth {collapsed}" {open}>
                <summary>Sign in with Ed25519 key</summary>
                <p>Run the following command and paste the result:</p>
                <code>hyprstream sign-challenge --nonce &lt;nonce&gt; --code-challenge &lt;cc&gt;</code>
                <p><a href="/oauth/authorize?{query}&prompt=local">Use Ed25519 challenge form</a></p>
            </details>"#,
            collapsed = collapsed,
            open = if providers.is_empty() { "open" } else { "" },
            query = authorize_query,
        )
    } else {
        String::new()
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Sign In — Hyprstream</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 480px; margin: 80px auto; padding: 0 20px; color: #333; }}
    h1 {{ font-size: 1.5em; text-align: center; }}
    .info {{ color: #666; font-size: 0.9em; text-align: center; margin-bottom: 2em; }}
    .provider-btn {{ display: block; padding: 12px 20px; margin: 10px 0;
                     background: #2563eb; color: white; text-align: center;
                     text-decoration: none; border-radius: 6px; font-size: 1em; }}
    .provider-btn:hover {{ background: #1d4ed8; }}
    .local-auth {{ margin-top: 2em; border-top: 1px solid #e5e7eb; padding-top: 1em; }}
    .local-auth summary {{ cursor: pointer; color: #666; }}
    .local-auth code {{ display: block; background: #f3f4f6; padding: 8px 12px;
                        border-radius: 4px; margin: 8px 0; font-size: 0.85em;
                        word-break: break-all; }}
    .local-auth a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <h1>Sign In</h1>
  <div class="info">
    <strong>{client_name}</strong> at {redirect_host} is requesting access.<br>
    Scopes: {scopes}
  </div>
  {provider_buttons}
  {local_section}
</body>
</html>"#,
        client_name = html_escape(client_name),
        redirect_host = html_escape(redirect_host),
        scopes = html_escape(scopes),
        provider_buttons = provider_buttons,
        local_section = local_section,
    )
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}
