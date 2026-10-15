# Troubleshooting

## Rate Limits

The API allows 100 requests per minute per token. Exceeding this returns HTTP 429 with
a `Retry-After` header giving the number of seconds to wait. The SDK retries 429
responses automatically up to three times with exponential backoff.

## Common Errors

A 401 means the token is missing, malformed, or expired. A 403 usually means the token
lacks the required scope. A 404 means the resource id does not exist. A 500 is a
server-side fault and is safe to retry.

## Debugging

Set `ACME_DEBUG=1` to log every request and response to stderr. Do not enable debug
logging in production because it prints tokens in plaintext.
