import vapoursynth as vs
from vapoursynth import core
import os

#core.std.LoadPlugin("vapoursynthplugins/libvsncnn.so")
core.std.LoadPlugin("vsort.dll")

clip = core.ffms2.Source(source=infile, cache=True, cachefile=os.path.basename(infile)+".ffindex")
clip_in = clip

matrix_in = clip.get_frame(0).props["_Matrix"]
width_in = clip.width
height_in = clip.height
pixels = width_in * height_in

if matrix_in == 2: # unknown
    if pixels > 720 * 576:
        matrix_in = 1 # bt.709
    else:
        matrix_in = 5 # bt.601

clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=matrix_in)
# Important: Clip out of range values. The model can't handle them.
clip = core.std.Levels(clip, min_in=0.0, max_in=1.0, min_out=0.0, max_out=1.0)

# clip = core.resize.Point(clip, height=height_in/2) # test bad deinterlacing

#clip = core.ncnn.Model([clip], '/mnt/exchange/net_g_50000_gan5_nomos.onnx', device_id=0, num_streams=2)
clip = core.ort.Model([clip], r'X:\span_ll63v2_net_g_490000.onnx', provider="DML", fp16=True)

matrix_out = 1 if pixels * 4 > 720 * 576 else 5

clip = core.resize.Bicubic(clip, width_in * 2, height_in * 2, format=vs.YUV420P8, matrix=matrix_out)

# A/B flicker test
if False:
    bicubic = core.resize.Bicubic(clip_in, width_in * 2, height_in * 2, format=vs.YUV420P8, matrix_in=matrix_in, matrix=matrix_out)
    #clip = core.std.Interleave([bicubic[0::2], clip[1::2]])
    clip = core.std.Interleave([bicubic[0::8], bicubic[1::8], bicubic[2::8], bicubic[3::8], clip[4::8], clip[5::8], clip[6::8], clip[7::8]])

clip.set_output()
