# HF Token and Speaker Model Cache Validation

## Goal

Make the WebUI's Hugging Face token and speaker-model download flow verifiable
and predictable. The page must inspect local model state on load, keep the
download action disabled for a complete cached model, and expose a separate
local cache-integrity check that the user can run at any time.

## Behavior

- HF tokens are never stored in browser `localStorage`.
- A submitted token is used for the current download and is persisted only in
  the existing encrypted local token store. The masked value `***` is never
  treated as a real token.
- Download token precedence is request token, environment token, then the
  encrypted local token.
- Model status checks are local-only and do not contact Hugging Face.
- A model is considered cached only when its expected local snapshot structure
  is present and contains usable model files. A download is successful only
  when this same check passes after the download call returns.
- The WebUI renders a disabled `已缓存` download button for complete models and
  a separate `检查缓存` action. Incomplete or missing models keep download
  available.
- Download errors are reported without token material and distinguish common
  authentication/permission, network/TLS, dependency, and generic failures.

## Data Flow

1. The WebUI requests `/api/speaker-models` and renders the local status.
2. `POST /api/speaker-models/{model_id}/download` accepts an optional form
   token, resolves the token using the precedence above, and downloads into
   the engine's existing cache directory.
3. The registry rechecks the cache after `snapshot_download`; an incomplete
   result becomes an HTTP error rather than a false success.
4. `GET /api/speaker-models/{model_id}/status` performs the same local check
   for the explicit `检查缓存` action.

## Testing

- Unit tests cover encrypted token round-trip and masked-token rejection.
- Registry tests cover complete/incomplete cache layouts, local-only status,
  token fallback, and download completeness enforcement.
- API tests cover token forwarding/persistence, error classification, and
  status responses without leaking token values.
- Browser verification covers initial status rendering, disabled cached
  buttons, explicit cache checks, and failed-download feedback.
