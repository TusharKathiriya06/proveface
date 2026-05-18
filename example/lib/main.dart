import 'package:flutter/material.dart';
import 'package:proveface/proveface.dart';

void main() => runApp(const ProveFaceExampleApp());

class ProveFaceExampleApp extends StatelessWidget {
  const ProveFaceExampleApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ProveFace Demo',
      theme: ThemeData.dark(),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({Key? key}) : super(key: key);
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  ProveFaceResult? _lastResult;

  Future<void> _startVerification() async {
    final result = await Navigator.push<ProveFaceResult>(
      context,
      MaterialPageRoute(
        builder: (_) => ProveFaceDetector(
          // Pass your app's colour here:
          // appBarColor: ColorConstants.BLUE_COLOR,
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
    if (!mounted) return;
    setState(() => _lastResult = result);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ProveFace Demo')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton.icon(
              onPressed: _startVerification,
              icon:  const Icon(Icons.face),
              label: const Text('Start Liveness Check'),
            ),
            const SizedBox(height: 32),
            if (_lastResult != null)
              _lastResult!.success
                  ? const Column(children: [
                      Icon(Icons.check_circle, color: Colors.green, size: 48),
                      SizedBox(height: 8),
                      Text('Verified ✓',
                        style: TextStyle(fontSize: 18, color: Colors.green)),
                    ])
                  : Column(children: [
                      const Icon(Icons.cancel, color: Colors.red, size: 48),
                      const SizedBox(height: 8),
                      Text(_lastResult!.errorMessage ?? 'Failed',
                        style: const TextStyle(color: Colors.red)),
                    ]),
          ],
        ),
      ),
    );
  }
}
