# Causal native selector integration

The frozen full-panel scheduler calls terminal_chunks with histories, weights,
horizon, path count and seed. Calendar-aware internal rolling CRPS also requires
the historical date prefix. This isolated scheduler clone binds that prefix into
a fresh request model before delegating to the original joint-asset forecast and
scoring implementation. It never passes target dates or future returns into
model fitting. It does not reuse a model choice from another portfolio.

The clone changes only the forecast import and its harness protocol identity.
Ordinary undated baseline forecasts delegate unchanged. Native model factories
bind the endpoint, whole-path or sequential selector, with 1,000 inner and final
paths. The manifest verifies 63 frozen parent source/configuration files; the
adapter also hashes its own and new selector source files into dependency
identity. Existing sealed dependency recovery provenance remains required.

Seven tests cover baseline loss parity, independent immutable date bindings,
invalid/unbound calls, observed prefix-only context, native endpoint/path/online
checkpoints and resume without new forecasts, source-tamper detection and the
exact two-line scheduler clone. The synthetic numerical fixture has 40 training
rows, four future rows and one asset. These checks do not establish real-panel
accuracy, 200-asset runtime or complete scheduler/registry integration. A real
panel smoke and source-bound registry admission remain necessary before the
final full suite. No existing scientific source or live worker is changed.
