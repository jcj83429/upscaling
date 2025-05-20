## 2xLiveActionV1_SPAN

**Scale:** 2
**Architecture:** SPAN
**Links:** https://github.com/jcj83429/upscaling/tree/main/4xLiveActionV1_SPAN

**Author:** jcj83429
**License:** CC BY-NC-SA 4.0
**Purpose:** Compression Removal, Deblur, Dehalo, General Upscaler
**Subject:** Video Frame
**Input Type:** Images
**Date:** 2025-05-19

**Size:** 48 channels
**I/O Channels:** 3(RGB)->3(RGB)

**Dataset:** nomosv2, expanded 6 times by downscaling and random brightness/contrast/colour changes
**Dataset Size:** 36000
**OTF (on the fly augmentations):** No
**Pretrained Model:** own pretrain on top of 2x_BHI_small_Redux_SPAN_S_1m30k (https://github.com/the-database/traiNNer-redux/releases/)
**Iterations:** 490000
**Batch Size:** 20
**GT Size:** 128

**Description:**
SPAN model for live action video and film. The main goal is to fix/reduce common video quality problems while maintaining fidelity. I tried the existing video-focused models and they all denoise or cause colour shifts so I decided to train my own.

The model is trained with compression (JPEG, MPEG-4 ASP, H264, VP9, H265), chroma subsampling, blurriness from multiple scaling, uneven horizontal and vertical resolution, oversharpening halos, bad deinterlacing jaggies, and onscreen text. It is not trained to remove noise at all so it preserves details in the source well. To prevent colour/brightness shifts, I used consistency loss in neosr. I had to modify consistency loss to use a stronger blur so it doesn't interfere with the halo removal.

Along with the model, I am also releasing my image degradation script. See github link. The script uses imagemagick and ffmpeg and it has a few unique features not found in other image degradation scripts: random text addition and sharpening with FIR (convolution) filters.

The random text addition adds random text with random font/colour/size/rotation/background. I found that it improves handling of screen content and hardsubs. Unfortunately, it has a hardcoded font list so it won't work as is on other people's computers.

The FIR sharpening can do different amount of sharpening in the vertical and horizontal directions. It can also produce very nasty oversharpening with secondary halos. I believe FIR sharpening closer to the kind of sharpening used in digital/electronic processes.

Limitations:
1. The model has limited ability to see details through heavy grain, but light to moderate grain is fine.
2. The model still does not handle bad deinterlacing perfectly, especially if the source is vertically resized. Fixing bad deinterlacing is not the main goal so it is what it is. Sources that are line-doubled throughout should be scaled back to half height first for best results.
3. The model sometimes oversharpens a little. This is because the training data has some oversharpened images.

**Showcase:** https://slow.pics/c/DtDN7gaq
