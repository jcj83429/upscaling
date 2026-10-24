!/bin/bash
find ../complexmotion -name '*.mkv' | while read line; do
    basename="`basename "$line"`"
    # force all clips to 12fps so they become 15fps (whole number) after telecine. The ivtc simulation is hardcoded to output 12fps.
    # Using 10fps results in skipped frames and ivtc cycle desync. Very weird.
    ffmpeg -fflags +genpts -r 12 -i "$line" -y \
    -c:v ffv1 -vf "format=yuv444p10le,scale=256:256:flags=area" "$basename" \
    -c:v ffv1 -vf "select='not(mod(n,2))',setpts=PTS/2,format=yuv444p10le,scale=256:256:flags=area" "$basename"-half.mkv \
    -c:v ffv1 -vf "select='not(mod(n,3))',setpts=PTS/3,format=yuv444p10le,scale=256:256:flags=area" "$basename"-third.mkv \
    -c:v ffv1 -vf "format=yuv444p10le,scale=256:256:flags=area,transpose=1" "$basename"-rot90.mkv \
    -c:v ffv1 -vf "select='not(mod(n,2))',setpts=PTS/2,format=yuv444p10le,scale=256:256:flags=area,transpose=1" "$basename"-half-rot90.mkv \
    -c:v ffv1 -vf "select='not(mod(n,3))',setpts=PTS/3,format=yuv444p10le,scale=256:256:flags=area,transpose=1" "$basename"-third-rot90.mkv \
    -c:v ffv1 -vf "format=yuv444p10le,scale=256:256:flags=area,rotate=PI:bilinear=0" "$basename"-rot180.mkv \
    -c:v ffv1 -vf "select='not(mod(n,2))',setpts=PTS/2,format=yuv444p10le,scale=256:256:flags=area,rotate=PI:bilinear=0" "$basename"-half-rot180.mkv \
    -c:v ffv1 -vf "select='not(mod(n,3))',setpts=PTS/3,format=yuv444p10le,scale=256:256:flags=area,rotate=PI:bilinear=0" "$basename"-third-rot180.mkv </dev/null
done

# Trim clips to multiple of 4 frames. This way, when selecting frames to use, the cumulative frame index across all clips can be used to tell which frames have residual combing (3rd frame of each 4 frame group)
ls *.mkv | while read line; do
    framecount="$(ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "$line")"
    newframecount="$(echo $framecount / 4 '*' 4 | bc)"
    ffmpeg -i "$line" -c copy -frames:v "$newframecount" "$line".mod4.mkv
    mv "$line".mod4.mkv "$line"
done
