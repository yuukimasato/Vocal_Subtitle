# HF Model Download and Token Storage Design

## Goal

After a user submits an HF Token, the WebUI must attempt to load the selected
pyannote model and only use diarization fallback when that load fails. A
successful token must be reusable without asking the user to paste it again.

## Current failure

The default installer does not include the `diarization` extra even though the
default configuration enables the pyannote backend. The download endpoint then
returns 503 because `pyannote.audio` cannot be imported. The global pipeline
also shares the embedding cache predicate, and the browser currently stores the
Token in `localStorage` as plain text.

## Design

1. Add `diarization` to the normal production/WebUI installation path so
   `torch` and `pyannote.audio` are present when the default backend is enabled.
2. Keep model download explicit. The embedding and global diarization endpoints
   resolve a submitted Token, load the model in a worker thread, and treat a
   successful load as the validation step. Missing optional dependencies remain
   a clear 503; authorization/protocol/network failures remain download errors.
3. Use a dedicated Hugging Face cache predicate for the global pipeline and
   verify the expected snapshot configuration before reporting success.
4. Store the Token server-side with Fernet authenticated encryption. Generate a
   per-installation key at `cache/.hf_token.key` with mode 0600 and store the
   encrypted value at `cache/hf_token.enc` with mode 0600. This protects the
   credential at rest from casual file inspection; it does not protect against
   a user who can read the running application and its key.
5. The browser stores only `***` as a presence marker. A masked submission asks
   the server to use the encrypted value. The clear Token is never returned by
   an API response or written to ordinary logs. The diarization and embedding
   engines also consult the encrypted store when no explicit Token is passed.

## Failure and fallback behavior

- Missing Token: HTTP 400 and no download attempt.
- Missing pyannote dependency: HTTP 503 with the install command.
- Model access denied, missing accepted agreement, invalid Token, or network
  failure: HTTP 502 with a safe diagnostic; the pipeline later records the
  failure and uses its configured fallback backend.
- Successful model load: persist the Token and return only model/status data.

## Testing

- Unit-test encryption round-trip, file permissions, and missing-store behavior.
- Test API resolution of a masked Token and persistence after a successful
  mocked download.
- Test that global cache detection is independent of embedding cache layout.
- Preserve existing tests for required Token, missing dependencies, and
  successful mocked downloads.

