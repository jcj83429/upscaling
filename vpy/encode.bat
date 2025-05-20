cd /D "%~dp0"
vspipe -c y4m upscale.vpy --arg infile=%1 - | ffmpeg -i %1 -i - -map 0 -map -0:v -map -0:t -map 1:v -c:a copy -sn -dn -shortest -crf 13 -preset veryfast -y "%~n1.sr.mkv"
pause