# FFmpeg Installation Guide

This directory should contain FFmpeg binaries for audio processing in the TTS system.

## Why FFmpeg is needed
FFmpeg is required for:
- Audio format conversion
- Audio resampling
- Audio stream processing
- Real-time audio manipulation

## Installation Methods

### Method 1: Download Static Build (Recommended for Linux)

```bash
# Navigate to this directory
cd streaming_tts/ffmpeg

# Download FFmpeg static build
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Extract
tar -xf ffmpeg-release-amd64-static.tar.xz

# Move binaries to current directory
mv ffmpeg-*-amd64-static/* .

# Clean up
rm -rf ffmpeg-*-amd64-static*
rm ffmpeg-release-amd64-static.tar.xz

# Verify installation
./bin/ffmpeg -version
```

### Method 2: System Package Manager

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg

# Create symbolic links to this directory
mkdir -p bin
ln -s /usr/bin/ffmpeg bin/ffmpeg
ln -s /usr/bin/ffprobe bin/ffprobe
```

#### CentOS/RHEL:
```bash
sudo yum install epel-release
sudo yum install ffmpeg

# Create symbolic links
mkdir -p bin
ln -s /usr/bin/ffmpeg bin/ffmpeg
ln -s /usr/bin/ffprobe bin/ffprobe
```

#### macOS:
```bash
brew install ffmpeg

# Create symbolic links
mkdir -p bin
ln -s /usr/local/bin/ffmpeg bin/ffmpeg
ln -s /usr/local/bin/ffprobe bin/ffprobe
```

### Method 3: Docker Environment

If using Docker, FFmpeg is already included in the container.
No additional installation needed.

## Directory Structure

After installation, this directory should contain:

```
streaming_tts/ffmpeg/
├── .gitkeep
├── README.txt
├── LICENSE
├── bin/
│   ├── ffmpeg          # Main FFmpeg binary
│   ├── ffprobe         # Media analysis tool
│   └── ...
├── doc/                # Documentation
└── presets/            # Encoding presets
```

## Verification

Test your FFmpeg installation:

```bash
# Check FFmpeg version
./bin/ffmpeg -version

# Test audio conversion
./bin/ffmpeg -f lavfi -i "sine=frequency=1000:duration=1" -ac 1 -ar 16000 test.wav

# Clean up test file
rm test.wav
```

## Troubleshooting

### Permission Issues
```bash
chmod +x bin/ffmpeg
chmod +x bin/ffprobe
```

### Path Issues
Make sure the binaries are in the correct location:
- `streaming_tts/ffmpeg/bin/ffmpeg`
- `streaming_tts/ffmpeg/bin/ffprobe`

### Missing Dependencies
If you get library errors, install required dependencies:

Ubuntu/Debian:
```bash
sudo apt install libavcodec-extra libavformat-dev libavutil-dev
```

### Alternative Download Sources

If the primary download fails, try these alternatives:

1. Official FFmpeg builds: https://ffmpeg.org/download.html
2. GitHub releases: https://github.com/BtbN/FFmpeg-Builds/releases
3. Zeranoe builds: https://www.gyan.dev/ffmpeg/builds/

## Notes

- FFmpeg binaries are excluded from git to reduce repository size
- Each user must install FFmpeg in this location
- The system will automatically detect FFmpeg in this directory
- For production deployment, consider using system-wide FFmpeg installation

## Support

For FFmpeg-related issues:
1. Check FFmpeg documentation: https://ffmpeg.org/documentation.html
2. Verify your installation with the verification steps above
3. Ensure proper file permissions
4. Check system dependencies

Last updated: 2025-07-23
