## 2xLiveActionHeavy_ESRGAN_lite

**Scale:** 2
**Architecture:** ESRGAN_lite
**Links:** https://github.com/jcj83429/upscaling/tree/main/2xLiveActionHeavy_ESRGAN_lite

**Author:** jcj83429
**License:** CC BY-NC-SA 4.0
**Purpose:** Compression Removal, Deblur, Dehalo, General Upscaler
**Subject:** Video Frame
**Input Type:** Images
**Date:** 2026-10-23

**Size:** 32 channels
**I/O Channels:** 3(RGB)->3(RGB)

**Dataset:** filtered DF2K + custom 4K bluray-based video tiles
**Dataset Size:** 50150
**OTF (on the fly augmentations):** No
**Pretrained Model:** own pretrain (3.8M iters)
**Iterations:** 375000
**Batch Size:** 16
**GT Size:** 256

**Description:**
Stronger upscaling model for digital live action video and film. The goal was the same as LiveActionV1 (to avoid colour shifts and keep as much details/texture/grain as possible). Compared to LiveActionV1, LiveActionHeavy has stronger deblurring, stronger dehalo and less oversharpening, but worse grain handling.

Like V1, the model is trained to handle compression (MPEG-2, MPEG-4 ASP, H264, VP9, H265), chroma subsampling, blurriness from multiple scaling, uneven horizontal and vertical resolution, oversharpening halos, bad deinterlacing jaggies (line doubling), and onscreen text. LiveActionHeavy is additionally trained with inter-frame compression artifacts, IVTC residual combing and matched noise in HR and LR.

Caveats:
1. Despite my best efforts to train the model to leave noise alone, it still sometimes amplifies or softens noise. Preserving grain and removing compression are conflicting goals.
2. This model is less temporally stable than V1 due to the increased sharpeness.

**Showcase:** https://slow.pics/c/aLHU19iP

### Training details

**Dataset Filtering**
The dataset I used for V1 doesn't work for bigger architectures. The problem is that downscaling an already tiled and filtered dataset (nomosv2) produces images that have excessive edge density, which causes uncontrollable noise amplification in the trained model. I had to start from scratch.

DIV2K is used as is. Flickr2K is filtered for JPEG artifacts using https://github.com/tinankh/GOD (image_filter_jpeg.py). Then, both DIV2K and Flickr2K are downscaled 50% with bilinear (triangle) filter (image_downscale.py) and tiled to 256x256 (image_tile.py).

The DF2K tiles are scored for focus with a custom difference-of-gaussians focus detector (image_dog.py). The focus detector is quite sensitive to noise so the absolute score is not meaningful. The idea is similar to filtering by IC9600 score but the difference is that the focus score doesn't penalize flat texture tiles as much, so it doesn't cause as much noise amplification as IC9600 filtering. I kept the top 60% and bottom ~1% of tiles by focus score. The lowest score tiles are actually just flat tiles. The intention was to reduce artifacts in flat areas.

The video tile dataset was sourced from video compression research datasets and 4K blurays. To pick the tiles, I made downscaled and compressed copies of the videos and ran them through a WIP version of the model that was not trained for inter-frame compression and manually picked the parts that had the worst artifacts. I made mpv_tile_helper.py to speed up the video tile creation. The video tiles were 1024x1024 and up to 2 seconds long. They are downscaled to 25% to reduce noise and blur, then rotated and framerate-converted to increase variety (derive.sh). Images made from the video tiles make up about 6.5% of the total training data. Because the video tile dataset is far lower quality than DF2K, it is important to use as little of of it as possible to keep the model sharp, hence the manual tile picking. I tried training with only the video tile dataset and the model came out much softer.

**Degradations**
For DF2K, the degradations (image_degrade_noise_compensation.py) are mostly the same as V1. The new changes are
1. Matched noise in HR and LR. This is the main feature and it took most of development time. More details below.
2. Mixed degradations. Two LRs with different degradations are generated and merged together with a mask. This improves the model's ability to remove DOF blur or localized deinterlacing jaggies. Consistent DOF preservation is not possible due to the limited receptive field so it's better to remove DOF to avoid the inconsistency of having halos of blur following sharp edges.
3. Edge mask for sharpening. Sharpening is only applied near edges. This trains the model not to soften the whole picture when removing oversharpening halos.
4. Stronger chroma subsampling. ESRGAN_lite needs this to sharpen chroma as well as SPAN.
5. Post-decode 1D resize (anamorphic decoding). The model still works best without post-decode resize though.

The matched noise works by generating a noise patch in HR resolution, adding it to the LR before compression, then adding the noise to the HR matched to the noise strength in the final decompressed LR. The noise strength matching is done by measuring the covariance between the original noise patch and the final LR image. Because compression and the degradations (the noise gets its own randomized degradation without mixed degradation) can sharpen or soften the noise, difference of gaussians is used to split the noise patch and the LR image into frequency ranges and the noise level matching is done per frequency range. Also, because inter-frame compression can affect noise differently than intra-frame compression, I actually generate a sequence of noise patches and a sequence of LR frames instead of only one frame to exercise inter-frame compression. A random frame is picked from the sequence to use as the training data. Finally, to help reduce grain pulsing caused by the model smoothing noise more in lower quality P and B frames, I measured the amount of noise kept by each codec and frame type (image_degrade_noise_gain_measure.py) and used this data to compensate by adding more noise to the HR for codecs and frame types that lose more noise.

For the video tiles, I used https://github.com/Kim2091/video-destroyer with H264, MPEG-2 and MPEG-4 ASP. I added hacks to add Telecine-IVTC simulation (video-destroyer.patch) to prevent the model from amplifying IVTC residual combing excessively. The hacks require the input to be 12fps. The actual mix of progressive and IVTC in the training data is not the same as what's writtin in the patch because I lost the original code used to generate the data. Due to the relatively small number of images, the video tile dataset is quite luck-dependent. Sometimes it works well and sometimes it doesn't depending on the random degradations that got rolled.

**Choice of Architecture**
I tried ESRGAN lite and RealPLKSR tiny. RealPLKSR Tiny was slower and had completely uncontrollable grain amplification even when training for full denoise. So ESRGAN it is.
For the discriminator, I ported ea2fpn from neosr to traiNNer-redux (put ea2fpn_arch.py in traiNNer-redux/traiNNer/archs/ to use). Even though it's unstable as hell and requires full FP32 precision, I find that it promotes sharpness better than dunet and metagan.

**Other Commentary**
This was intended to be a V2 but I couldn't get the grain handling to be as consistent as V1 so I decided to call it Heavy. In retrospect, it was a miracle that I was able to get V1 to work in less than 2 months. SPAN seems to reconstruct edges and focus its action on edges while leaving flat areas alone. This turned out to be really good for preserving grain and handling compression artifacts, which tend to be worst around edges. ESRGAN requires more explicit training for grain handling to be able to sharpen edges through grain and avoid amplifying grain. All the new changes I added to the degradation script are to train for things that SPAN does on its own. Unfortunately, once I started adding noise to the LR, I found it impossible to make the model preserve grain consistently because compression changes noise in complex ways so it is not possible to match the noise in LR and HR perfectly. Also, some video encoders like x264 turn noise into fast-moving compression artifacts that look fine in motion but look like a blocky mess when paused. A single frame architecture like ESRGAN probably can't know if the artifacts should be smoothed out as artifacts or kept as grain.
