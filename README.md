<div align="center">

# proveface

**Real-time face liveness detection for Flutter**

[![pub version](https://img.shields.io/pub/v/proveface.svg)](https://pub.dev/packages/proveface)
[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![platform](https://img.shields.io/badge/platform-android%20%7C%20ios-lightgrey)](#platform-support)

Blink · Smile · Head Turn · Anti-Spoofing · Permission Handled

</div>

---

## What is proveface?

`proveface` is a Flutter package that verifies a user is a **real, live person**
in front of the camera — not a photo, video, or screen replay.

It runs a randomised sequence of face challenges (blink, smile, head turns) and
simultaneously runs passive anti-spoofing in the background. When all challenges
pass, it returns a base64-encoded JPEG of the verified face.

---

## Features

| Feature | Details |
|---|---|
| 👁 Blink | Eyes-closed → eyes-open, multi-frame confirmed |
| 😊 Smile | Geometry-verified, 5-frame confirmation |
| ↔️ Head turn | Left, right, up, down with ghost-face guide overlay |
| 🛡 Passive anti-spoof | ML model scoring every 6th frame |
| 📺 Screen detection | Pixel texture + banding analysis |
| 📷 Quality checks | Distance, centering, lighting, glasses, mask detection |
| 🔐 Permission | Camera permission requested automatically |
| ⚙️ Configurable | Toggle each challenge, set individual timeouts |
| 📦 Typed result | `ProveFaceResult` with base64 JPEG on success |

---

## Platform support

| Android | iOS |
|---------|-----|
| ✅ API 23+ | ✅ iOS 12.0+ |

---

## Quick start

### 1. Add dependency

```yaml
dependencies:
  proveface: ^1.0.0
```

### 2. Android — `android/app/build.gradle`

```gradle
android {
  defaultConfig {
    minSdkVersion 23
  }
}
```

### 3. iOS — `ios/Runner/Info.plist`

```xml
<key>NSCameraUsageDescription</key>
<string>Used for face liveness verification.</string>
```

### 4. Use it

```dart
import 'package:proveface/proveface.dart';

final result = await Navigator.push<ProveFaceResult>(
  context,
  MaterialPageRoute(
    builder: (_) => ProveFaceDetector(
      appBarColor: const Color(0xFF0D4582),
      config: const ProveFaceConfig(
        enableBlink:     true,
        enableSmile:     true,
        enableTurnLeft:  true,
        enableTurnRight: true,
        enableTurnUp:    false,
        enableTurnDown:  false,
      ),
    ),
  ),
);

if (result?.success == true) {
  final base64Image = result!.capturedImageBase64!;
}
```

---

## How it works

```
User opens screen
       │
       ▼
Camera permission check ──(denied)──► Permission screen with "Open Settings"
       │
    (granted)
       ▼
Face quality checks
(distance · centering · lighting · glasses · mask)
       │
       ▼
Passive anti-spoof running in background
(ML model + screen texture analysis)
       │
       ▼
Randomised challenge sequence
(blink → smile → turn left → turn right)
       │
       ▼
All passed → Capture → Validate → Return ProveFaceResult
```

---

## Configuration

```dart
ProveFaceConfig(
  enableBlink:           true,
  enableSmile:           true,
  enableTurnLeft:        true,
  enableTurnRight:       true,
  enableTurnUp:          false,
  enableTurnDown:        false,
  blinkTimeoutSeconds:   7,
  smileTimeoutSeconds:   7,
  turnTimeoutSeconds:    7,
  sessionTimeoutSeconds: 0,    // 0 = auto
)
```

---

## Result

```dart
result.success               // true / false
result.capturedImageBase64   // base64 JPEG, non-null on success
result.errorMessage          // reason, non-null on failure
```

---

## Project structure

```
proveface/
├── lib/
│   ├── proveface.dart              ← single import
│   └── src/
│       ├── prove_face_detector.dart
│       ├── prove_face_config.dart
│       ├── prove_face_result.dart
│       └── face_check_status.dart
├── android/src/main/AndroidManifest.xml
├── example/lib/main.dart
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Contributing

This repository has branch protection on `main`.
All changes must go through a Pull Request — direct pushes are disabled.

1. Fork the repo
2. Create a branch: `git checkout -b fix/your-fix`
3. Commit your changes
4. Open a Pull Request

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

Built by [Tushar Kathiriya](https://github.com/TusharKathiriya06)