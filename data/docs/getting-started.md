# Getting Started

## Install

Install the SDK with pip:

```bash
pip install acme-sdk
```

The SDK requires Python 3.10 or newer. We recommend a virtual environment.

## Quickstart

Create a client and make your first request:

```python
from acme import Client
client = Client(token="ACME_TOKEN")
print(client.ping())
```

The `ping` method returns `"pong"` when the API is reachable.

## Configuration

The client reads configuration from environment variables. `ACME_TOKEN` is required;
`ACME_BASE_URL` defaults to `https://api.acme.example`. The default request timeout
is 30 seconds. You can raise it for slow batch jobs by passing `timeout=120` to the
constructor. Configuration values passed to the constructor override environment
variables, which in turn override built-in defaults.
