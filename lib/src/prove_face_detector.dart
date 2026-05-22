import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:ui' as ui;
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:face_anti_spoofing_detector/face_anti_spoofing_detector.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:permission_handler/permission_handler.dart';
import 'face_check_status.dart';
import 'prove_face_config.dart';
import 'prove_face_result.dart';

part 'prove_face_detector_android.dart';

// ─── Internal challenge enum ──────────────────────────────────────────────────

enum _Challenge { blink, smile, turnLeft, turnRight, turnUp, turnDown }

// ─── Widget ───────────────────────────────────────────────────────────────────

/// A full-screen face liveness detector widget.
///
/// Camera permission is requested automatically. Push as a route and await
/// a [ProveFaceResult]:
/// ```dart
/// final result = await Navigator.push<ProveFaceResult>(
///   context,
///   MaterialPageRoute(builder: (_) => const ProveFaceDetector()),
/// );
/// if (result?.success == true) {
///   sendToServer(result!.capturedImageBase64!);
/// }
/// ```
class ProveFaceDetector extends StatefulWidget {
  /// Controls which challenges are enabled and their timeouts.
  final ProveFaceConfig config;

  /// Optional callback fired with the result just before the route pops.
  final void Function(ProveFaceResult result)? onResult;

  /// AppBar title. Defaults to `'Face Verification'`.
  final String title;

  /// AppBar background colour.
  /// Defaults to `Color(0xFF0D4582)` — pass your own `ColorConstants.BLUE_COLOR` here.
  final Color appBarColor;

  const ProveFaceDetector({
    Key? key,
    this.config      = const ProveFaceConfig(),
    this.onResult,
    this.title       = 'Face Verification',
    this.appBarColor = const Color(0xFF0D4582),
  }) : super(key: key);

  @override
  State<ProveFaceDetector> createState() => Platform.isAndroid
      ? _ProveFaceDetectorAndroidState()
      : _ProveFaceDetectorState();
}

// ─── State ────────────────────────────────────────────────────────────────────

class _ProveFaceDetectorState extends State<ProveFaceDetector>
    with WidgetsBindingObserver {

  // ── Detection thresholds (private, not exposed to callers) ───────────────
  static const double _spoofThreshold              = 0.55;
  static const int    _spoofFrameWindow            = 8;
  static const double _eyeClosedThreshold          = 0.3;
  static const double _eyeOpenThreshold            = 0.7;
  static const double _smileThreshold              = 0.75;
  static const double _darkThreshold               = 40.0;
  static const double _brightThreshold             = 230.0;
  static const double _darkGlassThreshold          = 50.0;
  static const double _landmarkVisibilityThreshold = 0.80;
  static const double _maxYawDegrees               = 25.0;
  static const double _tooCloseThreshold           = 90.0;
  static const double _tooFarThreshold             = 40.0;
  static const double _turnYawThreshold            = 25.0;
  static const double _turnPitchThreshold          = 15.0;
  static const double _turnNeutralYaw              = 12.0;
  static const double _turnNeutralPitch            = 12.0;
  static const int    _turnExtremeFramesRequired   = 3;
  static const int    _turnNeutralFramesRequired   = 3;
  static const int    _angleBufferSize             = 4;
  static const double _screenTextureMinVariance    = 8.0;
  static const double _screenBandingMaxVariance    = 12.0;
  static const int    _smileConfirmFrames          = 5;
  static const int    _debounceFrames              = 3;
  static const int    _leaveReadyRequired          = 5;

  // ── ML Kit ────────────────────────────────────────────────────────────────
  late final FaceDetector _faceDetector;

  // ── Camera ────────────────────────────────────────────────────────────────
  CameraDescription? _camera;
  CameraController?  _cameraController;

  // ── Processing guards ─────────────────────────────────────────────────────
  bool _canProcess       = true;
  bool _isBusy           = false;
  bool _isCapturing      = false;
  bool _cameraInitializing = false;

  // ── Passive anti-spoof ────────────────────────────────────────────────────
  bool _spoofModelReady = false;
  final List<double> _spoofScores = [];

  // ── Challenge pool ────────────────────────────────────────────────────────
  List<_Challenge> _challengePool  = [];
  int              _challengeIndex = 0;
  int?             _trackedFaceId;

  // ── Blink state ───────────────────────────────────────────────────────────
  bool _blinkDone     = false;
  bool _wasEyesClosed = false;
  bool _blinkStarted  = false;
  bool _blinkTimedOut = false;
  Timer? _blinkTimer;

  // ── Smile state ───────────────────────────────────────────────────────────
  bool _smileDone       = false;
  bool _smileStarted    = false;
  bool _smileTimedOut   = false;
  int  _smileFrameCount = 0;
  Timer? _smileTimer;

  // ── Turn state ────────────────────────────────────────────────────────────
  bool _turnLeftDone   = false;
  bool _turnRightDone  = false;
  bool _turnUpDone     = false;
  bool _turnDownDone   = false;
  bool _wasTurnedLeft  = false;
  bool _wasTurnedRight = false;
  bool _wasTurnedUp    = false;
  bool _wasTurnedDown  = false;
  bool _turnStarted    = false;
  bool _turnTimedOut   = false;
  Timer? _turnTimer;
  int _turnExtremeFrames = 0;
  int _turnNeutralFrames = 0;
  final List<double> _yawBuffer   = [];
  final List<double> _pitchBuffer = [];

  // ── Face quality ──────────────────────────────────────────────────────────
  FaceCheckStatus _status        = FaceCheckStatus.noFace;
  FaceCheckStatus _pendingStatus = FaceCheckStatus.noFace;
  int _pendingFrames    = 0;
  int _leaveReadyFrames = 0;
  int _frameIdx         = 0;
  int _noFaceGraceCount = 0;
  static const int _noFaceGraceMax = 18;

  // ── Spoof bounding box ────────────────────────────────────────────────────
  Rect? _lastFaceBoundingBox;

  // ── Capture ───────────────────────────────────────────────────────────────
  bool    _isImageCaptured     = false;
  bool    _isUploading         = false;
  bool    _isWaitingToComplete = false;
  bool    _isValidatingCapture = false;
  bool    _errorInImage        = false;
  String  _errorMessage        = '';
  String? _capturedBase64;

  // ── Permission ────────────────────────────────────────────────────────────
  bool _permissionGranted = false;
  bool _permissionChecked = false;

  // ── Session / countdown ───────────────────────────────────────────────────
  Timer? _sessionTimer;
  int    _challengeSecondsLeft = 0;
  int    _sessionSecondsLeft   = 0;
  Timer? _uiTicker;

  ProveFaceConfig get _cfg => widget.config;

  // ─────────────────────────────────────────────────────────────────────────
  // LIFECYCLE
  // ─────────────────────────────────────────────────────────────────────────

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        enableContours:       true,
        enableLandmarks:      true,
        enableClassification: true,
        minFaceSize:          0.15,
        performanceMode:      FaceDetectorMode.accurate,
        enableTracking:       true,
      ),
    );
    _buildChallengePool();
    _requestCameraPermission();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized) { return; }
    if (state == AppLifecycleState.inactive) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _canProcess = false;
    _blinkTimer?.cancel();
    _smileTimer?.cancel();
    _turnTimer?.cancel();
    _sessionTimer?.cancel();
    _uiTicker?.cancel();
    _stopCamera();
    _faceDetector.close();
    FaceAntiSpoofingDetector.destroy();
    super.dispose();
  }

  // ─────────────────────────────────────────────────────────────────────────
  // PERMISSION
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _requestCameraPermission() async {
    final status = await Permission.camera.request();
    if (!mounted) return;
    if (status.isGranted) {
      setState(() {
        _permissionGranted = true;
        _permissionChecked = true;
      });
      _initPassiveModel();
      _initCamera();
    } else {
      setState(() {
        _permissionGranted = false;
        _permissionChecked = true;
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CHALLENGE POOL
  // ─────────────────────────────────────────────────────────────────────────

  void _buildChallengePool() {
    final pool = <_Challenge>[];
    if (_cfg.enableBlink)     { pool.add(_Challenge.blink); }
    if (_cfg.enableSmile)     { pool.add(_Challenge.smile); }
    if (_cfg.enableTurnLeft)  { pool.add(_Challenge.turnLeft); }
    if (_cfg.enableTurnRight) { pool.add(_Challenge.turnRight); }
    if (_cfg.enableTurnUp)    { pool.add(_Challenge.turnUp); }
    if (_cfg.enableTurnDown)  { pool.add(_Challenge.turnDown); }
    pool.shuffle(Random());
    _challengePool  = pool;
    _challengeIndex = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // INIT
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _initPassiveModel() async {
    try {
      final status = await FaceAntiSpoofingDetector.initialize();
      if (mounted) { setState(() => _spoofModelReady = status == true); }
    } catch (e) {
      debugPrint('[ProveFace] Passive spoof model init failed: $e');
    }
  }

  Future<void> _initCamera() async {
    if (_cameraInitializing) return;
    _cameraInitializing = true;
    try {
      final cameras = await availableCameras();
      _camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );
      _cameraController = CameraController(
        _camera!,
        ResolutionPreset.high,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid
            ? ImageFormatGroup.nv21
            : ImageFormatGroup.bgra8888,
      );
      await _cameraController!.initialize();
      if (!mounted) return;
      await _cameraController!.lockCaptureOrientation(DeviceOrientation.portraitUp);
      await _cameraController!.startImageStream(_onCameraImage);
      _startSessionTimer();
      _startUiTicker();
      setState(() {});
    } catch (e) {
      debugPrint('[ProveFace] Camera init failed: $e');
    } finally {
      _cameraInitializing = false;
    }
  }

  Future<void> _stopCamera() async {
    final ctrl = _cameraController;
    _cameraController = null;
    try { await ctrl?.stopImageStream(); } catch (_) {}
    try { await ctrl?.dispose(); } catch (_) {}
  }

  // ─────────────────────────────────────────────────────────────────────────
  // TIMERS
  // ─────────────────────────────────────────────────────────────────────────

  void _startSessionTimer() {
    _sessionTimer?.cancel();
    final timeout = _cfg.effectiveSessionTimeout;
    if (timeout <= 0) { return; }
    _sessionSecondsLeft = timeout;
    _sessionTimer = Timer(Duration(seconds: timeout), () {
      if (mounted && !_isCapturing) _restart(message: 'Session timed out. Please try again.');
    });
  }

  void _startUiTicker() {
    _uiTicker?.cancel();
    _uiTicker = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted) { return; }
      setState(() {
        if (_sessionSecondsLeft   > 0) { _sessionSecondsLeft--; }
        if (_challengeSecondsLeft > 0) { _challengeSecondsLeft--; }
      });
    });
  }

  void _startBlinkTimer() {
    _blinkTimer?.cancel();
    _challengeSecondsLeft = _cfg.blinkTimeoutSeconds;
    _blinkTimer = Timer(Duration(seconds: _cfg.blinkTimeoutSeconds), () {
      if (!_blinkDone && mounted) {
        setState(() => _blinkTimedOut = true);
        Future.delayed(const Duration(milliseconds: 1500), _restart);
      }
    });
  }

  void _startSmileTimer() {
    _smileTimer?.cancel();
    _challengeSecondsLeft = _cfg.smileTimeoutSeconds;
    _smileTimer = Timer(Duration(seconds: _cfg.smileTimeoutSeconds), () {
      if (!_smileDone && mounted) {
        setState(() => _smileTimedOut = true);
        Future.delayed(const Duration(milliseconds: 1500), _restart);
      }
    });
  }

  void _startTurnTimer() {
    _turnTimer?.cancel();
    _challengeSecondsLeft = _cfg.turnTimeoutSeconds;
    _turnTimer = Timer(Duration(seconds: _cfg.turnTimeoutSeconds), () {
      if (mounted) {
        setState(() => _turnTimedOut = true);
        Future.delayed(const Duration(milliseconds: 1500), _restart);
      }
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CAMERA STREAM
  // ─────────────────────────────────────────────────────────────────────────

  void _onCameraImage(CameraImage image) {
    if (!_canProcess || _isBusy || _isCapturing) { return; }
    _isBusy = true;
    _processFrame(image).whenComplete(() {
      _isBusy = false;
      if (mounted) setState(() {});
    });
  }

  Future<void> _processFrame(CameraImage image) async {
    _frameIdx++;

    final inputImage = _buildInputImage(image);
    if (inputImage == null) { return; }

    List<Face> faces;
    try {
      faces = await _faceDetector.processImage(inputImage);
    } catch (e) {
      debugPrint('[ProveFace] FaceDetector error: $e');
      return;
    }
    if (!_canProcess || !mounted) return;

    final _Challenge? activeTurn = _challengeIndex < _challengePool.length
        ? _challengePool[_challengeIndex]
        : null;
    final bool isTurnChallengeActive =
        activeTurn == _Challenge.turnLeft  ||
        activeTurn == _Challenge.turnRight ||
        activeTurn == _Challenge.turnUp    ||
        activeTurn == _Challenge.turnDown;

    if (faces.isEmpty) {
      // After completing at least one challenge, any face loss = full restart
      if (_challengeIndex > 0) {
        if (_isCapturing) return;
        _restart(message: 'Face not detected. Please restart verification.');
        return;
      }
      // First challenge started but not yet complete — short grace period
      if (_blinkStarted || _smileStarted || _turnStarted) {
        _noFaceGraceCount++;
        if (_noFaceGraceCount < _noFaceGraceMax) {
          if (isTurnChallengeActive && _turnStarted) {
            _commitStatus(_waitingStatusFor(activeTurn!));
          }
          return;
        }
      }
      _noFaceGraceCount = 0;
      _resetChallenges();
      _commitStatus(FaceCheckStatus.noFace);
      return;
    }
    _noFaceGraceCount = 0;

    if (faces.length > 1) {
      _resetChallenges();
      _commitStatus(FaceCheckStatus.multipleFaces);
      return;
    }

    final face      = faces.first;
    // ── CHANGE 1: use inputImageData (v0.9 API) instead of metadata (v0.11) ─
    final imageSize = inputImage.metadata!.size;

    final yaw   = _correctedYaw(face);
    final pitch = face.headEulerAngleX ?? 0.0;
    final roll  = face.headEulerAngleZ ?? 0.0;

    final distance = _calcFaceDistance(face);
    if (distance > _tooCloseThreshold) { _commitStatus(FaceCheckStatus.tooClose);  return; }
    if (distance < _tooFarThreshold && distance > 5) { _commitStatus(FaceCheckStatus.tooFar); return; }

    if (!_isFaceCentered(face, imageSize)) { _commitStatus(FaceCheckStatus.notCentered); return; }

    final bypassYaw   = activeTurn == _Challenge.turnLeft  || activeTurn == _Challenge.turnRight;
    final bypassPitch = activeTurn == _Challenge.turnUp    || activeTurn == _Challenge.turnDown;
    final yawViolation   = !bypassYaw   && yaw.abs()   > _maxYawDegrees;
    final pitchViolation = !bypassPitch && pitch.abs() > 20.0;
    if (yawViolation || pitchViolation || roll.abs() > 20.0) {
      _commitStatus(FaceCheckStatus.faceNotFrontal);
      return;
    }

    if (!isTurnChallengeActive) {
      final obstacleStatus = _checkObstacles(face);
      if (obstacleStatus != null) { _commitStatus(obstacleStatus); return; }

      final occlusionStatus = _checkFaceOcclusion(face);
      if (occlusionStatus != null) { _commitStatus(occlusionStatus); return; }
    }

    if (!isTurnChallengeActive && _frameIdx % 4 == 0) {
      final qualityStatus = await _checkImageQuality(image, face, imageSize);
      if (!_canProcess || !mounted) return;
      if (qualityStatus != null) { _commitStatus(qualityStatus); return; }
    }

    if (_frameIdx % 6 == 0 && _spoofModelReady && _lastFaceBoundingBox != null) {
      final spoofStatus = await _checkPassiveLiveness(image, face);
      if (!_canProcess || !mounted) return;
      if (spoofStatus != null) { _commitStatus(spoofStatus); return; }
    }

    if (_frameIdx % 8 == 0) {
      final screenStatus = _checkScreenArtifacts(image, face);
      if (screenStatus != null) { _commitStatus(screenStatus); return; }
    }

    final bool anyChallengeLive = _blinkStarted || _smileStarted || _turnStarted;

    // Before any challenge starts, track the current face.
    // Once a challenge is live, lock the ID and restart if it changes.
    if (face.trackingId != null) {
      if (!anyChallengeLive) {
        _trackedFaceId = face.trackingId;
      } else if (_trackedFaceId != null && face.trackingId != _trackedFaceId) {
        if (_isCapturing) return;
        _restart(message: 'Different face detected. Please restart verification.');
        return;
      }
    }

    while (_challengeIndex < _challengePool.length) {
      final s = _runChallenge(_challengePool[_challengeIndex], face, yaw, pitch);
      if (s != null) { _commitStatus(s); return; }
      _challengeIndex++;
    }

    _commitStatus(FaceCheckStatus.ready);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CHALLENGE RUNNERS
  // ─────────────────────────────────────────────────────────────────────────

  FaceCheckStatus? _runChallenge(_Challenge c, Face face, double yaw, double pitch) {
    switch (c) {
      case _Challenge.blink:     return _runBlinkChallenge(face);
      case _Challenge.smile:     return _runSmileChallenge(face);
      case _Challenge.turnLeft:  return _runTurnChallenge(c, yaw, pitch);
      case _Challenge.turnRight: return _runTurnChallenge(c, yaw, pitch);
      case _Challenge.turnUp:    return _runTurnChallenge(c, yaw, pitch);
      case _Challenge.turnDown:  return _runTurnChallenge(c, yaw, pitch);
      // ── CHANGE 2: default required in Dart 2 for non-void returning switch ─
      default:                   return null;
    }
  }

  FaceCheckStatus? _runBlinkChallenge(Face face) {
    if (_blinkDone) { return null; }
    if (!_blinkStarted) { _blinkStarted = true; _startBlinkTimer(); }
    if (_blinkTimedOut) { return FaceCheckStatus.blinkTimedOut; }
    _detectBlink(face);
    return _blinkDone ? null : FaceCheckStatus.waitingForBlink;
  }

  FaceCheckStatus? _runSmileChallenge(Face face) {
    if (_smileDone) { return null; }
    if (!_smileStarted) { _smileStarted = true; _startSmileTimer(); }
    if (_smileTimedOut) { return FaceCheckStatus.smileTimedOut; }
    _detectSmile(face);
    return _smileDone ? null : FaceCheckStatus.waitingForSmile;
  }

  FaceCheckStatus? _runTurnChallenge(_Challenge c, double yaw, double pitch) {
    if (_isTurnDone(c)) {
      _turnStarted       = false;
      _turnTimedOut      = false;
      _turnExtremeFrames = 0;
      _turnNeutralFrames = 0;
      _yawBuffer.clear();
      _pitchBuffer.clear();
      return null;
    }
    if (!_turnStarted) { _turnStarted = true; _startTurnTimer(); }
    if (_turnTimedOut) { return FaceCheckStatus.turnTimedOut; }
    _detectTurn(c, yaw, pitch);
    if (_isTurnDone(c)) { return null; }
    if (_isTurnExtremeReached(c)) { return FaceCheckStatus.returnToCenter; }
    return _waitingStatusFor(c);
  }

  bool _isTurnDone(_Challenge c) {
    switch (c) {
      case _Challenge.turnLeft:  return _turnLeftDone;
      case _Challenge.turnRight: return _turnRightDone;
      case _Challenge.turnUp:    return _turnUpDone;
      case _Challenge.turnDown:  return _turnDownDone;
      default:                   return false;
    }
  }

  bool _isTurnExtremeReached(_Challenge c) {
    switch (c) {
      case _Challenge.turnLeft:  return _wasTurnedLeft;
      case _Challenge.turnRight: return _wasTurnedRight;
      case _Challenge.turnUp:    return _wasTurnedUp;
      case _Challenge.turnDown:  return _wasTurnedDown;
      default:                   return false;
    }
  }

  FaceCheckStatus _waitingStatusFor(_Challenge c) {
    switch (c) {
      case _Challenge.turnLeft:  return FaceCheckStatus.waitingForTurnLeft;
      case _Challenge.turnRight: return FaceCheckStatus.waitingForTurnRight;
      case _Challenge.turnUp:    return FaceCheckStatus.waitingForTurnUp;
      case _Challenge.turnDown:  return FaceCheckStatus.waitingForTurnDown;
      default:                   return FaceCheckStatus.waitingForTurnLeft;
    }
  }

  void _detectBlink(Face face) {
    final l = face.leftEyeOpenProbability  ?? 1.0;
    final r = face.rightEyeOpenProbability ?? 1.0;
    if (l < _eyeClosedThreshold && r < _eyeClosedThreshold) { _wasEyesClosed = true; }
    if (_wasEyesClosed && l > _eyeOpenThreshold && r > _eyeOpenThreshold) {
      _blinkDone = true;
      _blinkTimer?.cancel();
    }
  }

  void _detectSmile(Face face) {
    final p = face.smilingProbability ?? 0.0;
    if (p > _smileThreshold && _isMouthGeometryValid(face)) {
      _smileFrameCount++;
      if (_smileFrameCount >= _smileConfirmFrames) {
        _smileDone = true;
        _smileTimer?.cancel();
      }
    } else {
      _smileFrameCount = 0;
    }
  }

  void _detectTurn(_Challenge c, double yaw, double pitch) {
    _yawBuffer.add(yaw);
    if (_yawBuffer.length > _angleBufferSize) { _yawBuffer.removeAt(0); }
    _pitchBuffer.add(pitch);
    if (_pitchBuffer.length > _angleBufferSize) { _pitchBuffer.removeAt(0); }

    final smoothYaw   = _yawBuffer.reduce((a, b)   => a + b) / _yawBuffer.length;
    final smoothPitch = _pitchBuffer.reduce((a, b) => a + b) / _pitchBuffer.length;

    void checkExtreme(bool alreadyReached, bool atExtreme, void Function() onConfirmed) {
      if (alreadyReached) { return; }
      if (atExtreme) {
        _turnExtremeFrames++;
        if (_turnExtremeFrames >= _turnExtremeFramesRequired) {
          _turnExtremeFrames = 0;
          onConfirmed();
        }
      } else {
        _turnExtremeFrames = 0;
      }
    }

    void checkNeutral(bool atNeutral, void Function() onConfirmed) {
      if (atNeutral) {
        _turnNeutralFrames++;
        if (_turnNeutralFrames >= _turnNeutralFramesRequired) {
          _turnNeutralFrames = 0;
          onConfirmed();
        }
      } else {
        _turnNeutralFrames = 0;
      }
    }

    switch (c) {
      case _Challenge.turnLeft:
        checkExtreme(_wasTurnedLeft, smoothYaw < -_turnYawThreshold, () => _wasTurnedLeft = true);
        if (_wasTurnedLeft) {
          checkNeutral(smoothYaw.abs() < _turnNeutralYaw, () { _turnLeftDone = true; _turnTimer?.cancel(); });
        }
        break;
      case _Challenge.turnRight:
        checkExtreme(_wasTurnedRight, smoothYaw > _turnYawThreshold, () => _wasTurnedRight = true);
        if (_wasTurnedRight) {
          checkNeutral(smoothYaw.abs() < _turnNeutralYaw, () { _turnRightDone = true; _turnTimer?.cancel(); });
        }
        break;
      case _Challenge.turnUp:
        checkExtreme(_wasTurnedUp, smoothPitch > _turnPitchThreshold, () => _wasTurnedUp = true);
        if (_wasTurnedUp) {
          checkNeutral(smoothPitch.abs() < _turnNeutralPitch, () { _turnUpDone = true; _turnTimer?.cancel(); });
        }
        break;
      case _Challenge.turnDown:
        checkExtreme(_wasTurnedDown, smoothPitch < -_turnPitchThreshold, () => _wasTurnedDown = true);
        if (_wasTurnedDown) {
          checkNeutral(smoothPitch.abs() < _turnNeutralPitch, () { _turnDownDone = true; _turnTimer?.cancel(); });
        }
        break;
      default: break;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // SCREEN / VIDEO DETECTION
  // ─────────────────────────────────────────────────────────────────────────

  FaceCheckStatus? _checkScreenArtifacts(CameraImage image, Face face) {
    try {
      final gray = _extractGrayFromCameraImage(image);
      if (gray == null) return null;
      final w  = image.width;
      final h  = image.height;
      final bb = face.boundingBox;
      final x0 = bb.left.toInt().clamp(0, w - 1);
      final y0 = bb.top.toInt().clamp(0, h - 1);
      final x1 = (bb.right.toInt()).clamp(0, w);
      final y1 = (bb.bottom.toInt()).clamp(0, h);
      if (x1 - x0 < 10 || y1 - y0 < 10) { return null; }

      double sum = 0, sumSq = 0; int count = 0;
      for (int row = y0; row < y1; row++) {
        for (int col = x0; col < x1; col++) {
          final v = gray[row * w + col].toDouble();
          sum += v; sumSq += v * v; count++;
        }
      }
      final mean     = sum / count;
      final variance = (sumSq / count) - (mean * mean);
      final stdDev   = sqrt(variance.clamp(0, double.infinity));
      if (stdDev < _screenTextureMinVariance) return FaceCheckStatus.screenDetected;

      final rowMeans = <double>[];
      for (int row = y0; row < y1; row++) {
        double rowSum = 0; int rowN = 0;
        for (int col = x0; col < x1 - 1; col++) {
          rowSum += (gray[row * w + col + 1] - gray[row * w + col]).abs().toDouble();
          rowN++;
        }
        if (rowN > 0) { rowMeans.add(rowSum / rowN); }
      }
      if (rowMeans.length < 4) { return null; }
      final rmMean = rowMeans.reduce((a, b) => a + b) / rowMeans.length;
      double rmVar = 0;
      for (final v in rowMeans) { rmVar += (v - rmMean) * (v - rmMean); }
      rmVar /= rowMeans.length;
      if (rmVar > _screenBandingMaxVariance) return FaceCheckStatus.screenDetected;
    } catch (e) {
      debugPrint('[ProveFace] Screen check error: $e');
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CHECK HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  FaceCheckStatus? _checkObstacles(Face face) {
    final bb = face.boundingBox;
    final visibleLandmarks = face.landmarks.values.where((lm) => lm != null).length;
    final visibilityRatio  = visibleLandmarks / FaceLandmarkType.values.length;
    if (visibilityRatio < _landmarkVisibilityThreshold) { return FaceCheckStatus.faceObstructed; }

    const requiredLm = [
      FaceLandmarkType.leftEye, FaceLandmarkType.rightEye,
      FaceLandmarkType.noseBase,
      FaceLandmarkType.leftMouth, FaceLandmarkType.rightMouth, FaceLandmarkType.bottomMouth,
    ];
    for (final lm in requiredLm) {
      if (face.landmarks[lm] == null) {
        if (lm == FaceLandmarkType.leftMouth  ||
            lm == FaceLandmarkType.rightMouth ||
            lm == FaceLandmarkType.bottomMouth) { return FaceCheckStatus.lowerFaceCovered; }
        if (lm == FaceLandmarkType.leftEye || lm == FaceLandmarkType.rightEye) {
          return FaceCheckStatus.eyesNotVisible;
        }
        return FaceCheckStatus.faceObstructed;
      }
    }

    final contours = face.contours;
    _lastFaceBoundingBox = bb;

    const requiredContours = [
      FaceContourType.face,
      FaceContourType.leftEye, FaceContourType.rightEye,
      FaceContourType.noseBridge, FaceContourType.noseBottom,
      FaceContourType.upperLipTop, FaceContourType.lowerLipBottom,
      FaceContourType.leftEyebrowTop, FaceContourType.rightEyebrowTop,
    ];
    const lipContours = {FaceContourType.upperLipTop, FaceContourType.lowerLipBottom};
    const eyeContours = {FaceContourType.leftEye, FaceContourType.rightEye};

    for (final ct in requiredContours) {
      final c = contours[ct];
      final minPts = lipContours.contains(ct) ? 5 : 1;
      if (c == null || c.points.length < minPts) {
        if (lipContours.contains(ct)) { return FaceCheckStatus.lowerFaceCovered; }
        if (eyeContours.contains(ct)) { return FaceCheckStatus.eyesNotVisible; }
        return FaceCheckStatus.faceObstructed;
      }
    }

    final upperLipC = face.contours[FaceContourType.upperLipTop]!;
    final lowerLipC = face.contours[FaceContourType.lowerLipBottom]!;
    final upperY = upperLipC.points.map((p) => p.y).reduce((a, b) => a + b) / upperLipC.points.length;
    final lowerY = lowerLipC.points.map((p) => p.y).reduce((a, b) => a + b) / lowerLipC.points.length;
    if ((lowerY - upperY).abs() < bb.height * 0.03) { return FaceCheckStatus.lowerFaceCovered; }

    final leftMouthLm   = face.landmarks[FaceLandmarkType.leftMouth]!;
    final rightMouthLm  = face.landmarks[FaceLandmarkType.rightMouth]!;
    final bottomMouthLm = face.landmarks[FaceLandmarkType.bottomMouth]!;
    final mouthRelY = (bottomMouthLm.position.y - bb.top) / bb.height;
    if (mouthRelY < 0.50 || mouthRelY > 0.95) { return FaceCheckStatus.lowerFaceCovered; }
    final mouthWidth = (rightMouthLm.position.x - leftMouthLm.position.x).abs();
    if (mouthWidth < bb.width * 0.20) { return FaceCheckStatus.lowerFaceCovered; }

    return _checkSunglasses(face, bb);
  }

  FaceCheckStatus? _checkSunglasses(Face face, Rect bb) {
    for (final eyeType in [FaceContourType.leftEye, FaceContourType.rightEye]) {
      final eyeContour = face.contours[eyeType];
      if (eyeContour == null || eyeContour.points.length < 4) { continue; }
      final xs = eyeContour.points.map((p) => p.x.toDouble()).toList();
      final ys = eyeContour.points.map((p) => p.y.toDouble()).toList();
      final cW = xs.reduce(max) - xs.reduce(min);
      final cH = ys.reduce(max) - ys.reduce(min);
      if (cW <= 0) { continue; }
      if (cH / cW < 0.12) { return FaceCheckStatus.sunglassesDetected; }
    }
    return null;
  }

  FaceCheckStatus? _checkFaceOcclusion(Face face) {
    final bb = face.boundingBox;
    final leftCheek  = face.landmarks[FaceLandmarkType.leftCheek];
    final rightCheek = face.landmarks[FaceLandmarkType.rightCheek];
    if (leftCheek == null || rightCheek == null) { return FaceCheckStatus.faceObstructed; }
    for (final cheek in [leftCheek, rightCheek]) {
      final relY = (cheek.position.y - bb.top) / bb.height;
      if (relY < 0.20 || relY > 0.62) { return FaceCheckStatus.faceObstructed; }
    }

    final noseLm = face.landmarks[FaceLandmarkType.noseBase];
    if (noseLm != null) {
      final relY = (noseLm.position.y - bb.top) / bb.height;
      final relX = (noseLm.position.x - bb.left) / bb.width;
      if (relY < 0.38 || relY > 0.68 || relX < 0.28 || relX > 0.72) {
        return FaceCheckStatus.faceObstructed;
      }
    }

    final leftEye  = face.landmarks[FaceLandmarkType.leftEye];
    final rightEye = face.landmarks[FaceLandmarkType.rightEye];
    if (leftEye != null && rightEye != null) {
      if ((leftEye.position.y - rightEye.position.y).abs() > bb.height * 0.15) {
        return FaceCheckStatus.faceObstructed;
      }
    }

    final faceContour = face.contours[FaceContourType.face];
    if (faceContour != null && faceContour.points.length >= 20) {
      final pts    = faceContour.points;
      final maxGap = bb.width * 0.13;
      for (int i = 0; i < pts.length; i++) {
        final curr = pts[i];
        final next = pts[(i + 1) % pts.length];
        final dx   = (curr.x - next.x).toDouble();
        final dy   = (curr.y - next.y).toDouble();
        if (sqrt(dx * dx + dy * dy) > maxGap) { return FaceCheckStatus.faceObstructed; }
      }
    }
    return null;
  }

  Future<FaceCheckStatus?> _checkImageQuality(
    CameraImage cameraImage, Face face, Size imageSize,
  ) async {
    try {
      final grayBytes = _extractGrayFromCameraImage(cameraImage);
      if (grayBytes == null) return null;
      final w  = cameraImage.width;
      final h  = cameraImage.height;
      final bb = face.boundingBox;
      final faceBrightness = _roiMean(
        grayBytes, w, h,
        bb.left.toInt(), bb.top.toInt(), bb.width.toInt(), bb.height.toInt(),
      );
      if (faceBrightness < _darkThreshold)   { return FaceCheckStatus.poorLighting; }
      if (faceBrightness > _brightThreshold) { return FaceCheckStatus.overExposed; }

      final leftEyeLm  = face.landmarks[FaceLandmarkType.leftEye];
      final rightEyeLm = face.landmarks[FaceLandmarkType.rightEye];
      if (leftEyeLm != null && rightEyeLm != null) {
        final eW = (bb.width  * 0.35).toInt();
        final eH = (bb.height * 0.20).toInt();
        final eY = (leftEyeLm.position.y - eH / 2).toInt();
        final lB = _roiMean(grayBytes, w, h, (leftEyeLm.position.x  - eW / 2).toInt(), eY, eW, eH);
        final rB = _roiMean(grayBytes, w, h, (rightEyeLm.position.x - eW / 2).toInt(), eY, eW, eH);
        if ((lB + rB) / 2 < _darkGlassThreshold) { return FaceCheckStatus.darkGlasses; }
      }
    } catch (e) {
      debugPrint('[ProveFace] Quality check error: $e');
    }
    return null;
  }

  Future<FaceCheckStatus?> _checkPassiveLiveness(CameraImage image, Face face) async {
    try {
      final Uint8List yuvBytes = _spoofInputBytes(image);
      final score = await FaceAntiSpoofingDetector.detect(
        yuvBytes:      yuvBytes,
        previewWidth:  image.width,
        previewHeight: image.height,
        orientation:   7,
        faceContour:   face.boundingBox,
      );
      if (score != null) {
        _spoofScores.add(score);
        if (_spoofScores.length > _spoofFrameWindow) { _spoofScores.removeAt(0); }
        if (_spoofScores.length >= 4) {
          final avg = _spoofScores.reduce((a, b) => a + b) / _spoofScores.length;
          if (avg < _spoofThreshold) return FaceCheckStatus.spoofDetected;
        }
      }
    } catch (e) {
      debugPrint('[ProveFace] Passive spoof check error: $e');
    }
    return null;
  }

  bool _isMouthGeometryValid(Face face) {
    final lm = face.landmarks[FaceLandmarkType.leftMouth];
    final rm = face.landmarks[FaceLandmarkType.rightMouth];
    if (lm == null || rm == null) { return false; }
    if ((rm.position.x - lm.position.x).abs() < face.boundingBox.width * 0.22) { return false; }
    final ul = face.contours[FaceContourType.upperLipTop];
    final ll = face.contours[FaceContourType.lowerLipBottom];
    if (ul == null || ll == null || ul.points.length < 5 || ll.points.length < 5) { return false; }
    final uY = ul.points.map((p) => p.y).reduce((a, b) => a + b) / ul.points.length;
    final lY = ll.points.map((p) => p.y).reduce((a, b) => a + b) / ll.points.length;
    return (lY - uY).abs() >= face.boundingBox.height * 0.03;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // RESET / RESTART
  // ─────────────────────────────────────────────────────────────────────────

  void _resetChallenges() {
    _blinkTimer?.cancel();
    _smileTimer?.cancel();
    _turnTimer?.cancel();
    _blinkDone = _wasEyesClosed = _blinkStarted = _blinkTimedOut = false;
    _smileDone = _smileStarted  = _smileTimedOut = false;
    _smileFrameCount = 0;
    _turnLeftDone = _turnRightDone = _turnUpDone   = _turnDownDone  = false;
    _wasTurnedLeft = _wasTurnedRight = _wasTurnedUp = _wasTurnedDown = false;
    _turnStarted       = _turnTimedOut = false;
    _turnExtremeFrames = _turnNeutralFrames = 0;
    _yawBuffer.clear();
    _pitchBuffer.clear();
    _noFaceGraceCount = 0;
    _leaveReadyFrames = 0;
    _spoofScores.clear();
    _challengeIndex = 0;
    _trackedFaceId  = null;
  }

  Future<void> _restart({String message = ''}) async {
    _canProcess = false;
    _blinkTimer?.cancel();
    _smileTimer?.cancel();
    _turnTimer?.cancel();
    _sessionTimer?.cancel();
    _uiTicker?.cancel();

    await _stopCamera();

    if (mounted) {
      setState(() {
        _canProcess  = true;
        _isBusy      = false;
        _isCapturing = false;
        _spoofScores.clear();

        _blinkDone = _wasEyesClosed = _blinkStarted = _blinkTimedOut = false;
        _smileDone = _smileStarted  = _smileTimedOut = false;
        _smileFrameCount = 0;

        _turnLeftDone  = _turnRightDone  = _turnUpDone   = _turnDownDone   = false;
        _wasTurnedLeft = _wasTurnedRight = _wasTurnedUp  = _wasTurnedDown  = false;
        _turnStarted   = _turnTimedOut = false;
        _turnExtremeFrames = _turnNeutralFrames = 0;
        _yawBuffer.clear();
        _pitchBuffer.clear();
        _noFaceGraceCount = 0;
        _trackedFaceId    = null;

        _leaveReadyFrames = 0;
        _frameIdx         = 0;
        _status           = FaceCheckStatus.noFace;
        _pendingStatus    = FaceCheckStatus.noFace;
        _pendingFrames    = 0;

        _isImageCaptured     = false;
        _isUploading         = false;
        _isWaitingToComplete = false;
        _isValidatingCapture = false;
        _errorInImage        = false;
        _errorMessage        = '';
        _capturedBase64      = null;

        _challengeSecondsLeft = 0;
        _sessionSecondsLeft   = 0;
      });
      _buildChallengePool();
    }

    await _initCamera();

    if (message.isNotEmpty) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) _showSnackBar(message, isError: true);
      });
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // STATUS DEBOUNCE
  // ─────────────────────────────────────────────────────────────────────────

  void _commitStatus(FaceCheckStatus s) {
    const alwaysImmediate = {
      FaceCheckStatus.noFace,
      FaceCheckStatus.multipleFaces,
      FaceCheckStatus.blinkTimedOut,
      FaceCheckStatus.smileTimedOut,
      FaceCheckStatus.turnTimedOut,
      FaceCheckStatus.returnToCenter,
    };

    if (alwaysImmediate.contains(s)) {
      _status = s; _pendingStatus = s; _pendingFrames = 0; _leaveReadyFrames = 0;
      return;
    }

    if (_status == FaceCheckStatus.ready) {
      if (s == FaceCheckStatus.ready) { _leaveReadyFrames = 0; return; }
      _leaveReadyFrames++;
      if (_leaveReadyFrames >= _leaveReadyRequired) {
        _status = s; _pendingStatus = s; _pendingFrames = _debounceFrames; _leaveReadyFrames = 0;
      }
      return;
    }

    _leaveReadyFrames = 0;
    if (s == _pendingStatus) {
      _pendingFrames++;
      if (_pendingFrames >= _debounceFrames) _status = s;
    } else {
      _pendingStatus = s; _pendingFrames = 1;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // CAPTURE
  // ─────────────────────────────────────────────────────────────────────────

  Future<void> _captureImage() async {
    if (_status != FaceCheckStatus.ready || _isCapturing) return;
    _sessionTimer?.cancel();
    _blinkTimer?.cancel();
    _smileTimer?.cancel();
    _turnTimer?.cancel();
    _uiTicker?.cancel();
    setState(() => _isCapturing = true);
    try {
      await _cameraController!.stopImageStream();
      final file      = await _cameraController!.takePicture();
      final bytes     = await File(file.path).readAsBytes();
      final base64Str = base64Encode(bytes);

      setState(() {
        _capturedBase64      = base64Str;
        _isImageCaptured     = true;
        _isValidatingCapture = true;
        _isWaitingToComplete = true;
      });

      final failReason = await _validateCapturedImage(file.path, bytes);
      if (!mounted) return;

      if (failReason != null) { _restart(message: failReason); return; }

      setState(() => _isValidatingCapture = false);
      await Future.delayed(const Duration(seconds: 1));
      if (mounted) {
        // ── CHANGE 3: typed result + optional callback instead of raw Map ──
        final result = ProveFaceResult.success(base64Str);
        widget.onResult?.call(result);
        Navigator.pop(context, result);
      }
    } catch (e) {
      debugPrint('[ProveFace] Capture error: $e');
      _restart(message: 'Error capturing image. Please try again.');
    }
  }

  Future<String?> _validateCapturedImage(String imagePath, Uint8List jpegBytes) async {
    try {
      // Confirm the file is fully written to disk before reading
      final file = File(imagePath);
      for (int i = 0; i < 5; i++) {
        if (await file.exists() && (await file.length()) > 0) break;
        await Future.delayed(const Duration(milliseconds: 100));
      }
      if (!await file.exists()) return 'Captured image not found. Please try again.';

      // Decode at stream-like resolution (640 px wide) so ML Kit behaves
      // consistently with the live stream — at full resolution ML Kit
      // estimates landmarks under occlusions, defeating the coverage checks.
      final codec = await ui.instantiateImageCodec(jpegBytes, targetWidth: 640);
      final frame = await codec.getNextFrame();
      final img   = frame.image;
      final valW  = img.width;
      final valH  = img.height;
      final byteData = await img.toByteData(format: ui.ImageByteFormat.rawRgba);
      img.dispose();
      codec.dispose();

      if (byteData == null) return 'Could not process captured image. Please try again.';

      final rgba       = byteData.buffer.asUint8List();
      final inputImage = _capturedFrameToInputImage(rgba, valW, valH);

      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty)    return 'No face detected in captured image. Please try again.';
      if (faces.length > 1) return 'Multiple faces in captured image. Please try again.';

      final face  = faces.first;
      final yaw   = face.headEulerAngleY ?? 0.0;
      final pitch = face.headEulerAngleX ?? 0.0;
      final roll  = face.headEulerAngleZ ?? 0.0;
      if (yaw.abs() > _maxYawDegrees || pitch.abs() > 20.0 || roll.abs() > 20.0) {
        return 'Face not looking straight. Please face the camera and try again.';
      }

      final obstacleStatus = _checkObstacles(face);
      if (obstacleStatus != null) return _statusMessageFor(obstacleStatus);

      final occlusionStatus = _checkFaceOcclusion(face);
      if (occlusionStatus != null) return _statusMessageFor(occlusionStatus);

      return null;
    } catch (e) {
      debugPrint('[ProveFace] Post-capture validation error: $e');
      return 'Could not validate captured image. Please try again.';
    }
  }

  String _statusMessageFor(FaceCheckStatus s) {
    switch (s) {
      case FaceCheckStatus.eyesNotVisible:     return 'Eyes not visible in captured image. Please remove any covering.';
      case FaceCheckStatus.darkGlasses:        return 'Dark glasses detected. Please remove them and try again.';
      case FaceCheckStatus.sunglassesDetected: return 'Sunglasses detected. Please remove them and try again.';
      case FaceCheckStatus.faceObstructed:     return 'Face is covered in captured image. Please remove any obstruction.';
      case FaceCheckStatus.lowerFaceCovered:   return 'Mask or face covering detected. Please remove it and try again.';
      case FaceCheckStatus.faceNotFrontal:     return 'Face not straight in captured image. Please try again.';
      default:                                 return 'Face check failed. Please try again.';
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // PIXEL UTILITIES
  // ─────────────────────────────────────────────────────────────────────────

  Uint8List? _extractGrayFromCameraImage(CameraImage image) {
    try {
      if (Platform.isAndroid) return image.planes.first.bytes;
      final bytes = image.planes.first.bytes;
      final gray  = Uint8List(image.width * image.height);
      for (int i = 0; i < gray.length; i++) {
        final o = i * 4;
        gray[i] = ((0.299 * bytes[o + 2]) + (0.587 * bytes[o + 1]) + (0.114 * bytes[o])).round();
      }
      return gray;
    } catch (_) { return null; }
  }

  double _roiMean(Uint8List gray, int imgW, int imgH, int x, int y, int w, int h) {
    x = x.clamp(0, imgW - 1);
    y = y.clamp(0, imgH - 1);
    w = (x + w).clamp(0, imgW) - x;
    h = (y + h).clamp(0, imgH) - y;
    if (w <= 0 || h <= 0) return 128.0;
    double sum = 0; int count = 0;
    for (int row = y; row < y + h; row++) {
      for (int col = x; col < x + w; col++) { sum += gray[row * imgW + col]; count++; }
    }
    return count > 0 ? sum / count : 128.0;
  }

  double _correctedYaw(Face face) => face.headEulerAngleY ?? 0.0;
  Uint8List _spoofInputBytes(CameraImage image) => image.planes.first.bytes;

  // Converts raw RGBA pixels (from ui.ImageByteFormat.rawRgba) into an
  // InputImage suitable for ML Kit. iOS uses BGRA8888; Android overrides
  // this in _ProveFaceDetectorAndroidState to produce NV21 instead.
  InputImage _capturedFrameToInputImage(Uint8List rgba, int width, int height) {
    final bgra = Uint8List(rgba.length);
    for (int i = 0; i < rgba.length; i += 4) {
      bgra[i]     = rgba[i + 2];
      bgra[i + 1] = rgba[i + 1];
      bgra[i + 2] = rgba[i];
      bgra[i + 3] = rgba[i + 3];
    }
    return InputImage.fromBytes(
      bytes: bgra,
      metadata: InputImageMetadata(
        size:        Size(width.toDouble(), height.toDouble()),
        rotation:    InputImageRotation.rotation0deg,
        format:      InputImageFormat.bgra8888,
        bytesPerRow: width * 4,
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // GEOMETRY UTILITIES
  // ─────────────────────────────────────────────────────────────────────────

  double _calcFaceDistance(Face face) {
    final le = face.landmarks[FaceLandmarkType.leftEye];
    final re = face.landmarks[FaceLandmarkType.rightEye];
    if (le == null || re == null) return 50.0;
    const double calibratedPD = 5.5;
    const double cameraAngle  = 22.5;
    final pd = sqrt(pow(le.position.x - re.position.x, 2) + pow(le.position.y - re.position.y, 2));
    return (pd / calibratedPD) / tan(cameraAngle * pi / 180);
  }

  bool _isFaceCentered(Face face, Size imageSize) {
    final bb = face.boundingBox;
    final dx = bb.center.dx - imageSize.width  / 2;
    final dy = bb.center.dy - imageSize.height / 2;
    return (dx * dx + dy * dy) <= 20000;
  }

  // ── CHANGE 4: InputImageData (v0.9 API) replaces InputImageMetadata (v0.11) ─

  InputImage? _buildInputImage(CameraImage image) {
    if (_cameraController == null || _camera == null) return null;
    const orientations = {
      DeviceOrientation.portraitUp:     0,
      DeviceOrientation.landscapeLeft:  90,
      DeviceOrientation.portraitDown:   180,
      DeviceOrientation.landscapeRight: 270,
    };
    final sensorOrientation = _camera!.sensorOrientation;
    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else {
      var comp = orientations[_cameraController!.value.deviceOrientation];
      if (comp == null) return null;
      comp = (sensorOrientation + comp) % 360;
      rotation = InputImageRotationValue.fromRawValue(comp);
    }
    if (rotation == null) return null;
    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21) ||
        (Platform.isIOS    && format != InputImageFormat.bgra8888)) return null;
    if (image.planes.length != 1) return null;
    final plane = image.planes.first;
    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size:        Size(image.width.toDouble(), image.height.toDouble()),
        rotation:    rotation,
        format:      format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  // ─────────────────────────────────────────────────────────────────────────
  // UI HELPERS
  // ─────────────────────────────────────────────────────────────────────────

  void _showSnackBar(String message, {bool isError = false}) {
    ScaffoldMessenger.of(context)
      ..removeCurrentSnackBar()
      ..showSnackBar(SnackBar(
        content: Text(message,
          textAlign: TextAlign.center,
          style: const TextStyle(color: Colors.white, fontSize: 13)),
        behavior:        SnackBarBehavior.floating,
        backgroundColor: isError ? Colors.red : Colors.green,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        margin: const EdgeInsets.only(bottom: 90, right: 20, left: 20),
      ));
  }

  _StatusUI get _statusUI {
    switch (_status) {
      case FaceCheckStatus.noFace:              return const _StatusUI('Position your face inside the circle',          Icons.face,                       false);
      case FaceCheckStatus.multipleFaces:       return const _StatusUI('Multiple faces detected. Please be alone.',     Icons.group_off,                   false);
      case FaceCheckStatus.tooClose:            return const _StatusUI('Too close — move back a little',                Icons.zoom_out,                    false);
      case FaceCheckStatus.tooFar:              return const _StatusUI('Too far — move closer',                         Icons.zoom_in,                     false);
      case FaceCheckStatus.notCentered:         return const _StatusUI('Centre your face in the circle',                Icons.center_focus_weak,           false);
      case FaceCheckStatus.faceNotFrontal:      return const _StatusUI('Look straight at the camera',                   Icons.rotate_90_degrees_ccw,      false);
      case FaceCheckStatus.eyesNotVisible:      return const _StatusUI('Eyes not visible — remove any covering',        Icons.visibility_off,              false);
      case FaceCheckStatus.darkGlasses:         return const _StatusUI('Please remove dark sunglasses',                 Icons.wb_sunny_outlined,           false);
      case FaceCheckStatus.sunglassesDetected:  return const _StatusUI('Sunglasses detected — please remove them',      Icons.remove_red_eye_outlined,     false);
      case FaceCheckStatus.faceObstructed:      return const _StatusUI('Face is covered — remove hand or any object',   Icons.back_hand_outlined,          false);
      case FaceCheckStatus.lowerFaceCovered:    return const _StatusUI('Please remove mask or face covering',           Icons.masks,                       false);
      case FaceCheckStatus.poorLighting:        return const _StatusUI('Too dark — move to a brighter area',            Icons.light_mode_outlined,         false);
      case FaceCheckStatus.overExposed:         return const _StatusUI('Too bright — avoid direct light behind you',    Icons.wb_sunny,                    false);
      case FaceCheckStatus.spoofDetected:       return const _StatusUI('Spoofing attempt detected — use your real face',Icons.no_photography_outlined,     false);
      case FaceCheckStatus.screenDetected:      return const _StatusUI('Screen or video detected — use your real face', Icons.tv_off_outlined,             false);
      case FaceCheckStatus.waitingForBlink:     return const _StatusUI('Please blink both eyes naturally',              Icons.remove_red_eye_outlined,     false);
      case FaceCheckStatus.waitingForSmile:     return const _StatusUI('Now give a natural smile 😊',                  Icons.sentiment_satisfied_alt,     false);
      case FaceCheckStatus.waitingForTurnLeft:  return const _StatusUI('Turn your head to the LEFT',                    Icons.arrow_back,                  false);
      case FaceCheckStatus.waitingForTurnRight: return const _StatusUI('Turn your head to the RIGHT',                   Icons.arrow_forward,               false);
      case FaceCheckStatus.waitingForTurnUp:    return const _StatusUI('Tilt your head UP',                             Icons.arrow_upward,                false);
      case FaceCheckStatus.waitingForTurnDown:  return const _StatusUI('Tilt your head DOWN',                           Icons.arrow_downward,              false);
      case FaceCheckStatus.returnToCenter:      return const _StatusUI('Good! Now look straight ahead',                 Icons.center_focus_strong,         false);
      case FaceCheckStatus.blinkTimedOut:       return const _StatusUI('Blink not detected in time. Restarting…',      Icons.timer_off,                   false);
      case FaceCheckStatus.smileTimedOut:       return const _StatusUI('Smile not detected in time. Restarting…',      Icons.timer_off,                   false);
      case FaceCheckStatus.turnTimedOut:        return const _StatusUI('Head turn not detected in time. Restarting…',  Icons.timer_off,                   false);
      case FaceCheckStatus.ready:               return const _StatusUI('Perfect! Tap the button to capture.',           Icons.check_circle_outline,        true);
      // ── CHANGE 5: default required for Dart 2 exhaustiveness ──────────────
      default:                                  return const _StatusUI('Position your face inside the circle',          Icons.face,                        false);
    }
  }

  bool get _isReady => _status == FaceCheckStatus.ready && !_errorInImage && !_isCapturing;

  _Challenge? get _activeTurnChallenge {
    if (_challengeIndex >= _challengePool.length) return null;
    final c = _challengePool[_challengeIndex];
    if (c == _Challenge.turnLeft  || c == _Challenge.turnRight ||
        c == _Challenge.turnUp    || c == _Challenge.turnDown) return c;
    return null;
  }

  _TimerData? get _sessionTimerData {
    if (_isWaitingToComplete || _isCapturing) return null;
    final timeout = _cfg.effectiveSessionTimeout;
    if (timeout > 0) return _TimerData('Session ends in', _sessionSecondsLeft, timeout);
    return null;
  }

  _TimerData? get _challengeTimerData {
    if (_isWaitingToComplete || _isCapturing) return null;
    if (_blinkStarted && !_blinkDone && _cfg.blinkTimeoutSeconds > 0) {
      return _TimerData('Blink in', _challengeSecondsLeft, _cfg.blinkTimeoutSeconds);
    }
    if (_smileStarted && !_smileDone && _cfg.smileTimeoutSeconds > 0) {
      return _TimerData('Smile in', _challengeSecondsLeft, _cfg.smileTimeoutSeconds);
    }
    if (_turnStarted && _cfg.turnTimeoutSeconds > 0) {
      return _TimerData('Turn in', _challengeSecondsLeft, _cfg.turnTimeoutSeconds);
    }
    return null;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // BUILD
  // ─────────────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    // ── Permission not yet checked — show loading ─────────────────────────
    if (!_permissionChecked) {
      return const Scaffold(
        backgroundColor: Color(0xFF0A1628),
        body: Center(
          child: CircularProgressIndicator(color: Color(0xFF0D4582)),
        ),
      );
    }

    // ── Permission denied — show friendly message with settings shortcut ──
    if (!_permissionGranted) {
      return Scaffold(
        backgroundColor: const Color(0xFF0A1628),
        appBar: AppBar(
          backgroundColor: widget.appBarColor,
          title: Text(widget.title,
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)),
          iconTheme: const IconThemeData(color: Colors.white),
          leading: IconButton(
            icon: const Icon(Icons.arrow_back_ios_new),
            onPressed: () => Navigator.of(context).pop(ProveFaceResult.failure('Camera permission denied')),
          ),
        ),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(32),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(Icons.no_photography_outlined, color: Color(0xFFFFD166), size: 64),
                const SizedBox(height: 20),
                const Text(
                  'Camera permission is required for face verification.',
                  textAlign: TextAlign.center,
                  style: TextStyle(color: Colors.white, fontSize: 16, height: 1.5),
                ),
                const SizedBox(height: 28),
                ElevatedButton.icon(
                  onPressed: () async {
                    await openAppSettings();
                    // re-check after returning from settings
                    await _requestCameraPermission();
                  },
                  icon:  const Icon(Icons.settings_outlined),
                  label: const Text('Open Settings'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFF0D4582),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    }

    // ── Normal flow ───────────────────────────────────────────────────────
    final statusUi    = _statusUI;
    final isReady     = _isReady;
    final sessionTd   = _sessionTimerData;
    final challengeTd = _challengeTimerData;
    final turnGuide   = _activeTurnChallenge;

    return WillPopScope(
      onWillPop: () async => !_isWaitingToComplete,
      child: Scaffold(
        backgroundColor: const Color(0xFF0A1628),
        appBar: AppBar(
          // ── CHANGE 6: ColorConstants.BLUE_COLOR replaced by widget.appBarColor ─
          backgroundColor: widget.appBarColor,
          title: Text(widget.title,
            style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w700)),
          iconTheme: const IconThemeData(color: Colors.white),
          leading: _isWaitingToComplete
              ? const SizedBox.shrink()
              : IconButton(
                  icon: const Icon(Icons.arrow_back_ios_new),
                  onPressed: () => Navigator.of(context).pop(
                    ProveFaceResult.failure('User cancelled'),
                  ),
                ),
        ),
        body: SafeArea(
          child: Stack(
            fit: StackFit.expand,
            children: [
              SingleChildScrollView(
                child: Column(
                  children: [
                    if (sessionTd != null)
                      Padding(
                        padding: const EdgeInsets.fromLTRB(24, 8, 24, 0),
                        child: _TimerRow(
                          label: sessionTd.label, seconds: sessionTd.seconds, total: sessionTd.total),
                      ),
                    Padding(
                      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 24),
                      child: _StepIndicator(pool: _challengePool, index: _challengeIndex),
                    ),
                    if (challengeTd != null)
                      Padding(
                        padding: const EdgeInsets.fromLTRB(24, 0, 24, 6),
                        child: _TimerRow(
                          label: challengeTd.label, seconds: challengeTd.seconds, total: challengeTd.total),
                      ),
                    const SizedBox(height: 4),
                    _CameraCircle(
                      controller:          _cameraController,
                      isReady:             isReady,
                      isCapturing:         _isCapturing || _isUploading,
                      capturedBase64:      _isImageCaptured ? _capturedBase64 : null,
                      activeTurnChallenge: turnGuide,
                      isTurnPhase2:        _status == FaceCheckStatus.returnToCenter,
                    ),
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
                      child: _StatusBanner(ui: statusUi, isReady: isReady),
                    ),
                    const SizedBox(height: 16),
                    if (_errorInImage)
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 20),
                        child: _ErrorPanel(
                          message: _errorMessage,
                          onRetry: () => _restart(),
                        ),
                      ),
                    if (!_errorInImage)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 32, top: 8),
                        child: _CaptureButton(
                          isReady:   isReady,
                          isLoading: _isUploading,
                          onTap:     _captureImage,
                        ),
                      ),
                  ],
                ),
              ),
              if (_isWaitingToComplete)
                Positioned.fill(
                  child: Container(
                    color: Colors.black.withOpacity(0.65),
                    child: Center(
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          CircularProgressIndicator(
                            color: _isValidatingCapture
                                ? const Color(0xFFFFD166)
                                : const Color(0xFF25C192),
                            strokeWidth: 3,
                          ),
                          const SizedBox(height: 20),
                          Text(
                            _isValidatingCapture
                                ? 'Validating captured image…'
                                : 'Verification successful!\nPlease wait…',
                            textAlign: TextAlign.center,
                            style: const TextStyle(
                              color: Colors.white, fontSize: 16,
                              fontWeight: FontWeight.w600, height: 1.5,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// INTERNAL DATA MODELS
// ─────────────────────────────────────────────────────────────────────────────

class _StatusUI {
  final String   message;
  final IconData icon;
  final bool     isSuccess;
  const _StatusUI(this.message, this.icon, this.isSuccess);
}

class _TimerData {
  final String label;
  final int    seconds;
  final int    total;
  const _TimerData(this.label, this.seconds, this.total);
}

// ─────────────────────────────────────────────────────────────────────────────
// SUB-WIDGETS (identical to your original file)
// ─────────────────────────────────────────────────────────────────────────────

class _TimerRow extends StatelessWidget {
  final String label;
  final int    seconds;
  final int    total;
  const _TimerRow({Key? key, required this.label, required this.seconds, required this.total})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    final fraction = total > 0 ? (seconds / total).clamp(0.0, 1.0) : 0.0;
    final Color barColor;
    if (fraction > 0.5) {
      barColor = const Color(0xFF25C192);
    } else if (fraction > 0.2) {
      barColor = const Color(0xFFFFD166);
    } else {
      barColor = const Color(0xFFE05252);
    }
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A2744),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(children: [
            Icon(Icons.timer_outlined, size: 14, color: barColor),
            const SizedBox(width: 6),
            Expanded(child: Text(label,
              style: TextStyle(color: barColor, fontSize: 12, fontWeight: FontWeight.w600))),
            Text('${seconds}s',
              style: TextStyle(color: barColor, fontSize: 13, fontWeight: FontWeight.w700)),
          ]),
          const SizedBox(height: 6),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value:            fraction,
              backgroundColor:  Colors.white12,
              valueColor:       AlwaysStoppedAnimation<Color>(barColor),
              minHeight:        4,
            ),
          ),
        ],
      ),
    );
  }
}

class _StepIndicator extends StatelessWidget {
  final List<_Challenge> pool;
  final int              index;
  const _StepIndicator({Key? key, required this.pool, required this.index}) : super(key: key);

  String _label(_Challenge c) {
    switch (c) {
      case _Challenge.blink:     return 'Blink';
      case _Challenge.smile:     return 'Smile';
      case _Challenge.turnLeft:  return 'Turn L';
      case _Challenge.turnRight: return 'Turn R';
      case _Challenge.turnUp:    return 'Turn U';
      case _Challenge.turnDown:  return 'Turn D';
      default:                   return '';
    }
  }

  @override
  Widget build(BuildContext context) {
    final allDone = index >= pool.length;
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          _Step(label: 'Position', done: true, active: index == 0 && pool.isNotEmpty),
          for (int i = 0; i < pool.length; i++) ...[
            _StepLine(done: i < index),
            _Step(label: _label(pool[i]), done: i < index, active: i == index),
          ],
          _StepLine(done: allDone),
          _Step(label: 'Capture', done: allDone, active: allDone),
        ],
      ),
    );
  }
}

class _Step extends StatelessWidget {
  final String label;
  final bool   done;
  final bool   active;
  const _Step({Key? key, required this.label, required this.done, required this.active})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 28, height: 28,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: done   ? const Color(0xFF25C192)
                 : active ? const Color(0xFF0D4582)
                          : Colors.white12,
            border: Border.all(
              color: active ? const Color(0xFF0D4582) : Colors.transparent, width: 2),
          ),
          child: Icon(done ? Icons.check : Icons.circle,
            size: 14, color: done || active ? Colors.white : Colors.white38),
        ),
        const SizedBox(height: 4),
        Text(label, style: TextStyle(
          fontSize: 10,
          color: done || active ? Colors.white : Colors.white38,
          fontWeight: active ? FontWeight.w700 : FontWeight.normal)),
      ],
    );
  }
}

class _StepLine extends StatelessWidget {
  final bool done;
  const _StepLine({Key? key, required this.done}) : super(key: key);

  @override
  Widget build(BuildContext context) => Container(
    width: 20, height: 2,
    margin: const EdgeInsets.only(bottom: 18),
    color: done ? const Color(0xFF25C192) : Colors.white12,
  );
}

class _CameraCircle extends StatelessWidget {
  final CameraController? controller;
  final bool              isReady;
  final bool              isCapturing;
  final String?           capturedBase64;
  final _Challenge?       activeTurnChallenge;
  final bool              isTurnPhase2;

  const _CameraCircle({
    Key? key,
    required this.controller,
    required this.isReady,
    required this.isCapturing,
    this.capturedBase64,
    this.activeTurnChallenge,
    this.isTurnPhase2 = false,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final borderColor = (isReady || isTurnPhase2)
        ? const Color(0xFF25C192)
        : const Color(0xFF0D4582);
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24),
      child: Container(
        decoration: BoxDecoration(
          shape:     BoxShape.circle,
          border:    Border.all(color: borderColor, width: 5),
          boxShadow: [BoxShadow(
            color: borderColor.withOpacity(0.35), blurRadius: 24, spreadRadius: 4)],
        ),
        child: controller?.value.isInitialized == true
            ? AspectRatio(
                aspectRatio: 1,
                child: ClipOval(
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Transform.scale(
                        scale: 2,
                        child: Center(
                          child: capturedBase64 != null
                              ? Image.memory(base64Decode(capturedBase64!), fit: BoxFit.cover)
                              : CameraPreview(controller!),
                        ),
                      ),
                      if (activeTurnChallenge != null)
                        CustomPaint(
                          painter: _TurnGuidePainter(
                              activeTurnChallenge!, phase2: isTurnPhase2),
                        ),
                    ],
                  ),
                ),
              )
            : AspectRatio(
                aspectRatio: 1,
                child: ClipOval(
                  child: Container(
                    color: Colors.black,
                    child: const Center(
                      child: CircularProgressIndicator(color: Color(0xFF0D4582))),
                  ),
                ),
              ),
      ),
    );
  }
}

class _StatusBanner extends StatelessWidget {
  final _StatusUI ui;
  final bool      isReady;
  const _StatusBanner({Key? key, required this.ui, required this.isReady}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 300),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: isReady ? const Color(0xFF0D3D2A) : const Color(0xFF1A2744),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: isReady
              ? const Color(0xFF25C192).withOpacity(0.6)
              : Colors.white12),
      ),
      child: Row(
        children: [
          Icon(ui.icon,
            color: isReady ? const Color(0xFF25C192) : const Color(0xFFFFD166), size: 22),
          const SizedBox(width: 12),
          Expanded(child: Text(ui.message, style: TextStyle(
            color:      isReady ? const Color(0xFF25C192) : Colors.white,
            fontSize:   14,
            fontWeight: FontWeight.w600,
          ))),
        ],
      ),
    );
  }
}

class _ErrorPanel extends StatelessWidget {
  final String        message;
  final VoidCallback  onRetry;
  const _ErrorPanel({Key? key, required this.message, required this.onRetry}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFF3D0A0A),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.red.withOpacity(0.5)),
      ),
      child: Column(
        children: [
          Text(message,
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.redAccent, fontSize: 14)),
          const SizedBox(height: 12),
          ElevatedButton.icon(
            onPressed: onRetry,
            icon:  const Icon(Icons.refresh, size: 18),
            label: const Text('Try Again'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF0D4582),
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
            ),
          ),
        ],
      ),
    );
  }
}

class _CaptureButton extends StatelessWidget {
  final bool         isReady;
  final bool         isLoading;
  final VoidCallback onTap;
  const _CaptureButton({Key? key, required this.isReady, required this.isLoading, required this.onTap})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size.width / 5;
    return GestureDetector(
      onTap: isReady && !isLoading ? onTap : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        width: size, height: size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: isReady ? const Color(0xFF0D4582) : Colors.white10,
          border: Border.all(
            color: isReady ? const Color(0xFF25C192) : Colors.white12, width: 4),
          boxShadow: isReady
              ? [BoxShadow(
                  color: const Color(0xFF25C192).withOpacity(0.4), blurRadius: 16)]
              : [],
        ),
        child: isLoading
            ? const Padding(
                padding: EdgeInsets.all(14),
                child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2.5))
            : Icon(Icons.camera_alt_rounded,
                size: size * 0.45,
                color: isReady ? Colors.white : Colors.white24),
      ),
    );
  }
}

class _TurnGuidePainter extends CustomPainter {
  final _Challenge direction;
  final bool       phase2;
  _TurnGuidePainter(this.direction, {this.phase2 = false});

  @override
  void paint(Canvas canvas, Size size) {
    final cx = size.width  / 2;
    final cy = size.height / 2;
    final r  = size.width  / 2;

    canvas.drawOval(
      Rect.fromCenter(center: Offset(cx, cy), width: r * 0.85, height: r * 1.1),
      Paint()
        ..color = (phase2 ? const Color(0xFF25C192) : Colors.white).withOpacity(0.15)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 1.5,
    );

    if (!phase2) {
      double dx = 0, dy = 0, scaleX = 1.0, scaleY = 1.0, tilt = 0;
      switch (direction) {
        case _Challenge.turnLeft:  dx = -r * 0.20; scaleX = 0.65; tilt =  0.30; break;
        case _Challenge.turnRight: dx =  r * 0.20; scaleX = 0.65; tilt = -0.30; break;
        case _Challenge.turnUp:    dy = -r * 0.18; scaleY = 0.72; tilt =  0.15; break;
        case _Challenge.turnDown:  dy =  r * 0.18; scaleY = 0.72; tilt = -0.15; break;
        default: break;
      }
      final ovalPaint = Paint()
        ..color = Colors.white.withOpacity(0.30)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0;
      final eyePaint = Paint()
        ..color = Colors.white.withOpacity(0.30)
        ..style = PaintingStyle.fill;
      canvas.save();
      canvas.translate(cx + dx, cy + dy);
      canvas.rotate(tilt);
      canvas.scale(scaleX, scaleY);
      canvas.drawOval(
        Rect.fromCenter(center: Offset.zero, width: r * 0.85, height: r * 1.1), ovalPaint);
      canvas.drawCircle(Offset(-r * 0.18, -r * 0.10), r * 0.045, eyePaint);
      canvas.drawCircle(Offset( r * 0.18, -r * 0.10), r * 0.045, eyePaint);
      canvas.restore();
    }

    final arrowColor = phase2
        ? const Color(0xFF25C192).withOpacity(0.95)
        : Colors.white.withOpacity(0.90);
    final arrowPaint = Paint()
      ..color = arrowColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.5
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    double ax, ay, bx, by;
    if (phase2) {
      switch (direction) {
        case _Challenge.turnLeft:  ax = cx - r * 0.72; ay = cy; bx = cx - r * 0.50; by = cy; break;
        case _Challenge.turnRight: ax = cx + r * 0.72; ay = cy; bx = cx + r * 0.50; by = cy; break;
        case _Challenge.turnUp:    ax = cx; ay = cy - r * 0.78; bx = cx; by = cy - r * 0.58; break;
        case _Challenge.turnDown:  ax = cx; ay = cy + r * 0.78; bx = cx; by = cy + r * 0.58; break;
        default: return;
      }
    } else {
      switch (direction) {
        case _Challenge.turnLeft:  ax = cx - r * 0.50; ay = cy; bx = cx - r * 0.72; by = cy; break;
        case _Challenge.turnRight: ax = cx + r * 0.50; ay = cy; bx = cx + r * 0.72; by = cy; break;
        case _Challenge.turnUp:    ax = cx; ay = cy - r * 0.58; bx = cx; by = cy - r * 0.78; break;
        case _Challenge.turnDown:  ax = cx; ay = cy + r * 0.58; bx = cx; by = cy + r * 0.78; break;
        default: return;
      }
    }

    canvas.drawLine(Offset(ax, ay), Offset(bx, by), arrowPaint);

    final angle = atan2(by - ay, bx - ax);
    const hw = 14.0;
    canvas.drawPath(
      Path()
        ..moveTo(bx, by)
        ..lineTo(bx - hw * cos(angle - pi / 6), by - hw * sin(angle - pi / 6))
        ..moveTo(bx, by)
        ..lineTo(bx - hw * cos(angle + pi / 6), by - hw * sin(angle + pi / 6)),
      arrowPaint,
    );
  }

  @override
  bool shouldRepaint(_TurnGuidePainter old) =>
      old.direction != direction || old.phase2 != phase2;
}
