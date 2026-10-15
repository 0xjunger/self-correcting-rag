# Authentication

## Tokens

Every request must include a bearer token in the `Authorization` header. Tokens are
issued from the dashboard and look like `acme_` followed by 32 hex characters.

## Scopes

Tokens carry scopes that limit what they can do. The available scopes are `read`,
`write`, and `admin`. A token with only `read` scope receives a 403 response on any
write endpoint. Scopes cannot be changed after a token is created; issue a new token
instead.

## Rotation

Rotate tokens at least every 90 days. The dashboard keeps the previous token valid for
a 24 hour grace period after rotation so in-flight jobs do not fail. After the grace
period the old token is rejected with a 401 response.
