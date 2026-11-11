# Native media qualification fixtures

These original procedural fixtures are distributed under the repository license.
They contain no external images, video, text or personal data.

Each PNG is a 128 by 128 RGB white canvas with a red square or blue circle drawn
inside `(24, 24, 104, 104)` using Pillow. Each MP4 contains four frames, encoded
with FFmpeg's `mpeg4` encoder at two frames per second in YUV420P.
`red-blue.mp4` contains two copies of the red square followed by two copies of
the blue circle; `blue-red.mp4` reverses that order. Encoded bytes are checked in
so decoder inputs and fingerprints are stable across hosts. The qualification
harness transports these encoded bytes through `/generate` and records SHA-256
fingerprints beside server evidence.
