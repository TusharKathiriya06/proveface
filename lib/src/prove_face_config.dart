/// Configuration for [ProveFaceDetector].
///
/// All challenges are enabled by default with 7-second timeouts.
///
/// Example — blink + smile only:
/// ```dart
/// ProveFaceConfig(
///   enableBlink:    true,
///   enableSmile:    true,
///   enableTurnLeft: false, enableTurnRight: false,
///   enableTurnUp:   false, enableTurnDown:  false,
/// )
/// ```
class ProveFaceConfig {
  final bool enableBlink;
  final bool enableSmile;
  final bool enableTurnLeft;
  final bool enableTurnRight;
  final bool enableTurnUp;
  final bool enableTurnDown;

  /// Set to `0` (default) to auto-calculate as the sum of all enabled
  /// challenge timeouts.
  final int sessionTimeoutSeconds;
  final int blinkTimeoutSeconds;
  final int smileTimeoutSeconds;
  final int turnTimeoutSeconds;

  const ProveFaceConfig({
    this.enableBlink           = true,
    this.enableSmile           = true,
    this.enableTurnLeft        = true,
    this.enableTurnRight       = true,
    this.enableTurnUp          = true,
    this.enableTurnDown        = true,
    this.sessionTimeoutSeconds = 0,
    this.blinkTimeoutSeconds   = 7,
    this.smileTimeoutSeconds   = 7,
    this.turnTimeoutSeconds    = 7,
  });

  int get effectiveSessionTimeout {
    if (sessionTimeoutSeconds > 0) return sessionTimeoutSeconds;
    int total = 0;
    if (enableBlink)     total += blinkTimeoutSeconds;
    if (enableSmile)     total += smileTimeoutSeconds;
    if (enableTurnLeft)  total += turnTimeoutSeconds;
    if (enableTurnRight) total += turnTimeoutSeconds;
    if (enableTurnUp)    total += turnTimeoutSeconds;
    if (enableTurnDown)  total += turnTimeoutSeconds;
    return total > 0 ? total : 30;
  }
}
