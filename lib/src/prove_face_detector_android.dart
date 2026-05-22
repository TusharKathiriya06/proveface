part of 'prove_face_detector.dart';

class _ProveFaceDetectorAndroidState extends _ProveFaceDetectorState {

  @override
  double _correctedYaw(Face face) {
    // Android front-camera sensor space is not pre-mirrored unlike iOS,
    // so negate yaw to match the user's left/right perspective.
    if (_camera?.lensDirection == CameraLensDirection.front) {
      return -(face.headEulerAngleY ?? 0.0);
    }
    return face.headEulerAngleY ?? 0.0;
  }

  @override
  bool _isFaceCentered(Face face, Size imageSize) {
    final bb = face.boundingBox;
    final dx = bb.center.dx - imageSize.width  / 2;
    final dy = bb.center.dy - imageSize.height / 2;
    return (dx * dx + dy * dy) <= 200000;
  }

  @override
  Uint8List _spoofInputBytes(CameraImage image) {
    // Some devices pack all NV21 data in a single plane; others split Y and VU.
    if (image.planes.length == 1) return image.planes[0].bytes;
    return _buildNV21Strided(image);
  }

  @override
  InputImage? _buildInputImage(CameraImage image) {
    if (_cameraController == null || _camera == null) return null;
    const orientations = {
      DeviceOrientation.portraitUp:     0,
      DeviceOrientation.landscapeLeft:  90,
      DeviceOrientation.portraitDown:   180,
      DeviceOrientation.landscapeRight: 270,
    };
    final sensorOrientation = _camera!.sensorOrientation;
    var comp = orientations[_cameraController!.value.deviceOrientation];
    if (comp == null) return null;
    comp = (sensorOrientation + comp) % 360;
    final rotation = InputImageRotationValue.fromRawValue(comp);
    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null || format != InputImageFormat.nv21) return null;
    if (image.planes.isEmpty) return null;

    final Uint8List bytes;
    final int bytesPerRow;
    if (image.planes.length == 1) {
      // Packed single-plane NV21 (most common on Android)
      bytes      = image.planes[0].bytes;
      bytesPerRow = image.planes[0].bytesPerRow;
    } else {
      // Separate Y + VU planes — normalise strides before passing to ML Kit
      bytes      = _buildNV21Strided(image);
      bytesPerRow = image.width;
    }

    return InputImage.fromBytes(
      bytes: bytes,
      metadata: InputImageMetadata(
        size:        Size(image.width.toDouble(), image.height.toDouble()),
        rotation:    rotation,
        format:      InputImageFormat.nv21,
        bytesPerRow: bytesPerRow,
      ),
    );
  }

  @override
  InputImage _capturedFrameToInputImage(Uint8List rgba, int width, int height) {
    final nv21 = _rgbaToNV21(rgba, width, height);
    return InputImage.fromBytes(
      bytes: nv21,
      metadata: InputImageMetadata(
        size:        Size(width.toDouble(), height.toDouble()),
        rotation:    InputImageRotation.rotation0deg,
        format:      InputImageFormat.nv21,
        bytesPerRow: width,
      ),
    );
  }

  Uint8List _rgbaToNV21(Uint8List rgba, int width, int height) {
    // NV21 UV plane covers floor(height/2) rows. If height is odd the last
    // even row index equals height-1 which exceeds floor(height/2) rows —
    // guard with evenH so we never write past the allocated UV region.
    final evenH = height & ~1;
    final nv21  = Uint8List(width * height + width * (evenH >> 1));
    int yIdx  = 0;
    int uvIdx = width * height;
    for (int row = 0; row < height; row++) {
      for (int col = 0; col < width; col++) {
        final i = (row * width + col) * 4;
        final r = rgba[i];
        final g = rgba[i + 1];
        final b = rgba[i + 2];
        nv21[yIdx++] = (((66 * r + 129 * g + 25 * b + 128) >> 8) + 16).clamp(0, 255);
        if (row < evenH && row % 2 == 0 && col % 2 == 0) {
          nv21[uvIdx++] = ((((112 * r - 94 * g - 18 * b + 128) >> 8) + 128)).clamp(0, 255);
          nv21[uvIdx++] = ((((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128)).clamp(0, 255);
        }
      }
    }
    return nv21;
  }

  // Copies Y and UV planes row-by-row to honour bytesPerRow padding.
  Uint8List _buildNV21Strided(CameraImage image) {
    final yPlane  = image.planes[0];
    final uvPlane = image.planes[1];
    final w = image.width;
    final h = image.height;

    final nv21 = Uint8List(w * h + (w * (h ~/ 2)));

    // Y plane — copy row by row stripping any padding
    int dstOffset = 0;
    for (int row = 0; row < h; row++) {
      final srcOffset = row * yPlane.bytesPerRow;
      nv21.setRange(dstOffset, dstOffset + w, yPlane.bytes, srcOffset);
      dstOffset += w;
    }

    // UV plane (interleaved VU in NV21) — copy row by row
    final uvHeight = h ~/ 2;
    for (int row = 0; row < uvHeight; row++) {
      final srcOffset = row * uvPlane.bytesPerRow;
      nv21.setRange(dstOffset, dstOffset + w, uvPlane.bytes, srcOffset);
      dstOffset += w;
    }

    return nv21;
  }
}
