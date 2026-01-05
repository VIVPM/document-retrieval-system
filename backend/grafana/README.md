# Grafana Cloud provisioning

Reproducible setup for this app's Grafana observability — a dashboard, two alert
rules, an email contact point, and a notification route — all built on the
`chat_messages_total` counter that `observability.py` exports.

Ported from ecommerce-flipkart-agent. The one panel change: flipkart's "messages
by tool" is gone (this app has no query router), replaced by "messages by
status" — the only label this counter carries.

## Files
- `dashboard.json` — "Document Retrieval System — Overview", 4 panels (message
  rate by status, success rate, errors, messages by status). Importable in the
  UI as-is (Dashboards → Import → paste), or pushed by the script.
- `provision.py` — creates the folder, dashboard, `drs-email` contact point,
  two alert rules, and appends a notification route.

## Apply
Needs these in `backend/.env` (NOT set yet — the OTLP push creds already there
are a *different* credential):
`GRAFANA_URL` (your stack, e.g. `https://<stack>.grafana.net`), `GRAFANA_API_TOKEN`
(a `glsa_` service-account token — **not** the OTLP push auth), `GRAFANA_ALERT_EMAIL`,
`GRAFANA_PROM_UID` (default `grafanacloud-prom`).

```bash
cd backend
python grafana/provision.py --dry-run   # print every payload, send nothing
python grafana/provision.py --apply      # create it all in Grafana
```

Until those are set, the dashboard still works: copy `dashboard.json` and paste
it into Grafana → Dashboards → Import.

## Alerts — provisioned but SILENT
Muted so they never email (the "no messages" rule is noisy for a low-traffic
demo). `provision.py` sets:
- `CREATE_ALERTS = True` — the two rules + email contact point + route are created.
- `MUTE_ALERTS = True` — an always-on mute timing is attached to the route, so the
  rules still evaluate and show in the UI but send no notifications.

The rules:
- **no messages (30m)** — fires when nothing is processed in 30m or the series is absent.
- **message errors** — fires when any message fails (`status="error"`) in 10m.

To actually receive emails, set `MUTE_ALERTS = False` and re-run `--apply`.
To tear the whole thing down (rules + mute timing + route + contact point):
```bash
python grafana/provision.py --remove-alerts   # dashboard stays
```
`--apply` is idempotent — existing rules/contact point are skipped, so re-running
only adds/updates the mute.

## Shared-stack safety
If this Grafana stack is shared with other projects, folder, dashboard, contact
point and alert rules are all additive. The notification policy is the one shared
object, and it's read-modify-**write**: the script fetches the existing tree and
appends our route (idempotently), never replacing it. `--dry-run` shows exactly
what it will do first.
