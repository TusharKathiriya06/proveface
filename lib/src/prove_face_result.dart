/// Result returned by [ProveFaceDetector] when it completes or is dismissed.
class ProveFaceResult {
  /// `true` if all liveness challenges passed and an image was captured.
  final bool success;

  /// Base64-encoded JPEG of the captured face. Non-null when [success] is `true`.
  final String? capturedImageBase64;

  /// Human-readable reason. Non-null when [success] is `false`.
  final String? errorMessage;

  const ProveFaceResult._({
    required this.success,
    this.capturedImageBase64,
    this.errorMessage,
  });

  factory ProveFaceResult.success(String base64Image) =>
      ProveFaceResult._(success: true, capturedImageBase64: base64Image);

  factory ProveFaceResult.failure([String? reason]) =>
      ProveFaceResult._(success: false, errorMessage: reason);

  @override
  String toString() => success
      ? 'ProveFaceResult(success, imageLength=${capturedImageBase64?.length})'
      : 'ProveFaceResult(failure, "$errorMessage")';
}
