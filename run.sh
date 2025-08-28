#!/usr/bin/env bash
video_dir="/large/ego4d/preprocessed/video"
feature_dir="/large/ego4d/preprocessed/video_feature"

for f in $(find $video_dir -name "*.mp4"); do
    base="$(basename "$f" .mp4)"
    if [ -f "${feature_dir}/${base}.pt" ]; then
        echo "Skipping $f (feature already exists)"
        continue
    fi
    echo "Processing $f"
    ffmpeg -i "$f" -c:v libx264 -preset veryfast -crf 23 -c:a copy -b:a 128k -y "tmp.mp4" && \
    mv "tmp.mp4" "$f"
done
uv run src/preprocess/preprocess_videos.py

ffmpeg -i /large/ego4d/preprocessed/video/7c3584ca-c6c7-41e8-a948-e6ff44d9e506_70.mp4 -c:v libx264 -preset veryfast -crf 23 -c:a copy -b:a 128k -y "tmp.mp4" && mv "tmp.mp4" "/large/ego4d/preprocessed/video/7c3584ca-c6c7-41e8-a948-e6ff44d9e506_70.mp4"