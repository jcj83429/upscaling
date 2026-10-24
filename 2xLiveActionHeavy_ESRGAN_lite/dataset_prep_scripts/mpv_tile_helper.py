#!/usr/bin/env python3
import mpv # https://github.com/jaseg/python-mpv
import subprocess
import sys
import os
from pathlib import Path

# capture settings
TILE_SIZE = 1024
TILE_DURATION = 2
OUT_DIR = '/home/archuser/video_tiles'
HDR_TONEMAP = 'zscale=transfer=linear,tonemap=tonemap=hable:desat=0,zscale=transfer=709:primaries=709:matrix=709,format=yuv420p10le'

# globals
filename = sys.argv[1]

res_x = TILE_SIZE
res_y = TILE_SIZE
is_hdr = False
box_pos_x = 0
box_pos_y = 0
current_time = 0

def my_log(loglevel, component, message):
    print('[{}] {}: {}'.format(loglevel, component, message))

player = mpv.MPV(log_handler=my_log, input_default_bindings=True, input_vo_keyboard=True)

# Property access, these can be changed at runtime
@player.property_observer('time-pos')
def time_observer(_name, value):
    global current_time
    if value is None:
        return
    # Here, _value is either None if nothing is playing or a float containing
    # fractional seconds since the beginning of the file.
    # print('Now playing at {:.2f}s'.format(value))
    current_time = value

#player.fullscreen = True
#player.loop_playlist = 'inf'
# Option access, in general these require the core to reinitialize
player['hwdec'] = 'nvdec-copy'
player['osc'] = 'yes'

def update_box_vf():
    player['vf'] = f'drawbox=x={box_pos_x}:y={box_pos_y}:width={TILE_SIZE}:height={TILE_SIZE}:color=red'

@player.property_observer('video-params/w')
def res_w_observer(_name, value):
    global res_x
    res_x = value

@player.property_observer('video-params/h')
def res_h_observer(_name, value):
    global res_y
    res_y = value

@player.property_observer('video-params/max-luma')
def hdr_observer(_name, value):
    global is_hdr
    is_hdr = value is not None and value > 100
    print("is hdr:", is_hdr)

@player.on_key_press('i')
def box_up():
    global box_pos_y
    box_pos_y = max(0, box_pos_y - 100)
    update_box_vf()

@player.on_key_press('k')
def box_down():
    global box_pos_y
    box_pos_y = min(res_y - TILE_SIZE, box_pos_y + 100)
    update_box_vf()

@player.on_key_press('j')
def box_left():
    global box_pos_x
    box_pos_x = max(0, box_pos_x - 100)
    update_box_vf()

@player.on_key_press('l')
def box_right():
    global box_pos_x
    box_pos_x = min(res_x - TILE_SIZE, box_pos_x + 100)
    update_box_vf()

@player.on_key_press('o')
def capture():
    outfilename = OUT_DIR + os.sep + Path(os.path.basename(filename)).stem + f'-{int(current_time)}.mkv'
    ffmpeg_cmd = ['ffmpeg', '-hwaccel', 'auto', '-ss', str(current_time), '-i', filename, '-vf', f'crop={TILE_SIZE}:{TILE_SIZE}:{box_pos_x}:{box_pos_y}' + ((','+HDR_TONEMAP) if is_hdr else ''), '-t', str(TILE_DURATION), '-map_metadata', '-1', '-an', '-sn', '-c:v', 'ffv1', '-y', outfilename]
    print(ffmpeg_cmd)
    subprocess.Popen(ffmpeg_cmd)

update_box_vf()

player.play(sys.argv[1])
player.wait_for_playback()

del player
