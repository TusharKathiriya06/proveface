## 1.0.2

### Bug Fixes
* Fixed fatal crash on Android when a camera frame was processed concurrently
  with `takePicture()`, causing `CameraDevice was already closed` exception.
* Fixed `CameraException: buildPreview() was called on a disposed CameraController`
  when session timed out.
* Fixed post-capture validation always failing on Android due to unsupported
  BGRA8888 format; Android now converts captured frame to NV21 for ML Kit.
* Fixed `RangeError` in NV21 conversion when decoded image has an odd height
  (e.g. portrait photos decoded at 640 px wide).

### Improvements
* Android camera processing extracted into a dedicated subclass for cleaner
  platform separation.
* Challenge countdown timer is now always visible once a challenge starts,
  regardless of transient face-quality interruptions.
* Face swap mid-challenge is now detected and triggers an immediate restart.
* Face loss after completing one or more challenges triggers an instant restart
  instead of waiting for a grace period.
* Post-capture image validation now detects face obstructions (hand, mask, etc.)
  consistently on both platforms by processing at stream-equivalent resolution.
* When all challenge checks are disabled in config, the widget proceeds directly
  to capture without requiring any liveness checks.

## 1.0.1

* Fixed: OSI-approved MIT license added for pub.dev recognition.
* Fixed: Static analysis lint rules configured in analysis_options.yaml.

## 1.0.0

* Initial release.
* Blink, smile, turn-left, turn-right, turn-up, turn-down liveness challenges.
* Randomised challenge ordering on every session.
* Passive anti-spoofing via `face_anti_spoofing_detector` ML model.
* Screen and video replay detection via pixel texture analysis.
* Face quality checks: distance, centering, lighting, sunglasses, mask.
* Camera permission requested automatically — no setup needed in consuming app.
* Permission denied screen with "Open Settings" shortcut.
* Typed `ProveFaceResult` with base64 JPEG on success.
* Configurable per-challenge timeouts via `ProveFaceConfig`.
* `appBarColor` and `title` customisable from call site.
* Supports Android API 23+ and iOS 12.0+.