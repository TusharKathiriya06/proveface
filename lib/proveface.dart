/// ProveFace — Flutter face liveness detection package.
///
/// Single import gives you everything:
/// ```dart
/// import 'package:proveface/proveface.dart';
/// ```
///
/// Then push [ProveFaceDetector] as a route:
/// ```dart
/// final result = await Navigator.push<ProveFaceResult>(
///   context,
///   MaterialPageRoute(builder: (_) => const ProveFaceDetector()),
/// );
/// if (result?.success == true) {
///   final base64Image = result!.capturedImageBase64!;
/// }
/// ```
library proveface;

export 'src/prove_face_detector.dart';
export 'src/prove_face_config.dart';
export 'src/prove_face_result.dart';
export 'src/face_check_status.dart';
