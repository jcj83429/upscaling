#!/usr/bin/python3
from PIL import Image
import sys
import os
import pdb
import random
import subprocess
import multiprocessing.pool
import tqdm
import string
from scipy import signal
import numpy as np

FACTOR = 2
HQ_SCALE = False
HQ_COMP = True
TEXT = True

characters = string.ascii_letters + string.digits + string.punctuation
# remove symbols that need escaping
characters = characters.replace('%', '').replace('@', '').replace('\\', '')

inDir = sys.argv[1]
parentDir, inDirBaseName = os.path.split(inDir.rstrip('/'))
outDirLR = os.path.join(parentDir, inDirBaseName+f'_LR_{FACTOR}x_degraded')
outDirHR = os.path.join(parentDir, inDirBaseName+f'_LR_{FACTOR}x_gt')
try:
    os.mkdir(outDirLR)
except FileExistsError:
    pass
try:
    os.mkdir(outDirHR)
except FileExistsError:
    pass

def resizeOptions(inX, inY):
    assert inX % FACTOR == 0
    assert inY % FACTOR == 0
    outX = inX // FACTOR
    outY = inY // FACTOR

    # Although the model is focused on low quality blurry/haloey sources, if not trained with high quality sharp sources it can handle them very poorly.
    cleanDigitalDownscale = HQ_SCALE or random.randrange(2) == 0

    finalFilterChoices = ['Triangle', 'Catrom', 'Lanczos', 'Spline']
    if cleanDigitalDownscale:
        # Box is less common but it produces more aliasing so it is useful to add some samples.
        # It is only used with one-step integer ratio scaling to avoid rounding weirdness.
        finalFilterChoices += ['Box']
    finalFilter = random.choice(finalFilterChoices)

    if finalFilter == 'Box':
        # Box is only done digitally in low quality software so it is always gamma light.
        useLinearLight = False
    else:
        # Downscaling with linear light expands bright areas. Downscaling with gamma light expands dark areas.
        # Neither extreme looks fully correct. Train with a mix of both to get a compromise.
        # Because box (10% probability) is always gamma, the overall mix is 63% linear 37% gamma
        useLinearLight = random.choices([True, False], [7, 3])[0]

    linearLightFlags = ['-colorspace', 'RGB']
    gammaLightFlags = ['-colorspace', 'sRGB']

    name = ''
    flags = []

    if useLinearLight:
        name += 'l-'
        flags += linearLightFlags
    else:
        name += 'g-'

    if not cleanDigitalDownscale:
        # Unsharp is more commonly seen on film and photo.
        # Electronic/digital FIR filter is more commonly seen on video, and it can achieve a level of nastiness that unsharp cannot.
        sharpeningMode = random.choices([None, 'unsharp', 'fir'], [2, 1, 1])[0]

        intermediateFilter = random.choice(['Triangle', 'Catrom', 'Lanczos', 'Spline', 'Jinc'])
        isInterlaced = random.randrange(5) == 0
        downX = random.randrange(outX * 3 // 5, outX * 5 // 4)
        if isInterlaced:
            interlaceFilter = random.choice(['Lanczos', 'Catrom', 'Triangle', 'Box'])
            if random.randrange(3) == 0:
                downY = outY // 2 # just bad deinterlacing, no rescaling
            else:
                downY = random.randrange(outY * 2 // 5, outY * 4 // 5)
            # Focus on point. Bilinear and Bicubic are not very different from multiple scaling
            deintFilter = random.choices(['Point', 'Triangle', 'Catrom'], [2, 1, 1])[0]
        else:
            downY = random.randrange(outY * 3 // 4, outY * 5 // 4)

        if isInterlaced:
            name += f'{intermediateFilter}-{interlaceFilter}-{downX}x{downY*2}i-'
            flags += ['-filter', intermediateFilter, '-resize', f'{downX}x{downY*2}!', '-filter', interlaceFilter, '-resize', f'{downX}x{downY}!']
        else:
            name += f'{intermediateFilter}-{downX}x{downY}-'
            flags += ['-filter', intermediateFilter, '-resize', f'{downX}x{downY}!']

        if sharpeningMode or isInterlaced:
            # Always sharpen in gamma. Sharpening in linear light produces extremely nasty dark halos.
            # Some real sources are actually like that but it is too extreme.
            # Deinterlacing is assumed to be in gamma light.
            if useLinearLight:
                flags += gammaLightFlags

            if sharpeningMode == 'unsharp':
                radius = random.choice([1,2,3])
                sigma = random.randrange(1, 50 - radius * 10) / 10
                name += f'unsharp{radius}x{sigma}-'
                flags += ['-sharpen', f'{radius}x{sigma}']
            elif sharpeningMode == 'fir':
                def randFilterParams():
                    taps = random.choice([3, 5])
                    freq = random.randrange(30, 46) / 100 if taps == 5 else 0.45
                    boost = random.randrange(11, 25) / 10
                    return taps, freq, boost
                # 2D seperable FIR filter
                hTaps, hFreq, hBoost = randFilterParams()
                hCoeffs = signal.firwin2(hTaps, [0, hFreq, hFreq + 0.1, 1], [1, 1, hBoost, hBoost], window=None)
                hCoeffs /= sum(hCoeffs) # must normalize to sum to 1 so the brightness doesn't change
                vTaps, vFreq, vBoost = randFilterParams()
                vCoeffs = signal.firwin2(vTaps, [0, vFreq, vFreq + 0.1, 1], [1, 1, vBoost, vBoost], window=None)
                vCoeffs /= sum(vCoeffs)
                vCoeffs = np.atleast_2d(vCoeffs).T # transpose
                kernel = vCoeffs * hCoeffs
                kernelStr = '   '.join(','.join(f'{i:f}' for i in row) for row in kernel)
                name += f'firsharp{hTaps}_{hFreq}_{hBoost}x{vTaps}_{vFreq}_{vBoost}-'
                flags += ['-morphology', 'Convolve', f'{hTaps}x{vTaps}: {kernelStr}']

            if isInterlaced:
                name += f'{deintFilter}-'
                flags += ['-filter', deintFilter, '-resize', f'{downX}x{downY*2}!']

            if useLinearLight:
                flags += linearLightFlags

    name += f'{finalFilter}'
    flags += ['-filter', finalFilter, '-resize', f'{outX}x{outY}!']

    if useLinearLight:
        flags += gammaLightFlags # output gamma light

    return name, flags

# Text addition is a part of the degrade script because text may need to be added before or after degrade
def textOptions(inX, inY, minSize):
    randomString = "".join([random.choice(characters) for i in range(50)])

    maxSize = 72
    # choose smaller sizes more often
    invSize = random.uniform(1 / maxSize, 1 / minSize)
    size = int(1 / invSize)

    angle = random.randrange(-360, 360)
    if angle < -90 or angle > 90:
        angle = 0 # 75% chance of upright text
    angle = angle % 360

    offsetX = random.randrange(-inX//2, inX//2)
    offsetY = random.randrange(-inY//2, inY//2)

    # random extra space between characters
    kerning = random.uniform(0, size) / 10

    fonts = [
        # sans serif
        # ttf-dejavu
        'DejaVu-Sans',
        # ttf-ibm-plex
        'IBM-Plex-Sans',
        'IBM-Plex-Sans-Bold',
        # ttf-liberation
        'Liberation-Sans',
        'Liberation-Sans-Bold',
        'NimbusMonoPS-Regular', # Courier
        'NimbusMonoPS-Bold',
        'NimbusSans-Regular', # Helvetica
        'NimbusSans-Bold',
        'NimbusSansNarrow-Regular',
        'NimbusSansNarrow-Bold',
        # noto-fonts
        'Noto-Sans-Regular',
        'Noto-Sans-Bold',
        # ttf-ubuntu-font-family
        'Ubuntu-Sans',
        'Ubuntu-Sans-Bold',
        # gsfonts
        'URWGothic-Book', # AvantGarde
        'URWGothic-Demi',
        # ttf-mscorefonts-installer
        'Arial',
        'Arial-Black',
        'Arial-Bold',
        'Comic-Sans-MS',
        'Comic-Sans-MS-Bold',
        'Impact',
        # serif
        # gsfonts
        'C059-Roman', # NewCenturySchlbk
        'C059-Bold',
        'NimbusRoman-Regular', # Times
        'NimbusRoman-Bold',
        'P052-Roman', # Palatino
        'P052-Bold',
        'URWBookman-Light',
        'URWBookman-Demi',
        # ttf-dejavu
        'DejaVu-Serif',
        'DejaVu-Serif-Bold',
        # ttf-ibm-plex
        'IBM-Plex-Serif',
        'IBM-Plex-Serif-Bold',
        # ttf-liberation
        'Liberation-Serif',
        'Liberation-Serif-Bold',
        # noto-fonts
        'Noto-Serif-Regular',
        'Noto-Serif-Bold',
    ]
    italicFonts = [
        # sans serif
        'NimbusMonoPS-BoldItalic', # Courier
        'NimbusMonoPS-Italic',
        'NimbusSans-BoldItalic', # Helvetica
        'NimbusSansNarrow-Oblique',
        'NimbusSans-Italic',
        'IBM-Plex-Sans-Italic',
        'IBM-Plex-Sans-Bold-Italic',
        'Liberation-Sans-Italic',
        'Liberation-Sans-Bold-Italic',
        'NimbusSans-Italic',
        'NimbusSans-BoldItalic',
        'Noto-Sans-Italic',
        'Noto-Sans-Bold-Italic',
        'Ubuntu-Sans-Italic',
        'Ubuntu-Sans-Bold-Italic',
        'URWGothic-BookOblique',
        'URWGothic-DemiOblique',
        'Arial-Italic',
        'Arial-Bold-Italic',
        # serif
        'C059-Italic', # NewCenturySchlbk
        'C059-Bold-Italic',
        'IBM-Plex-Serif-Italic',
        'IBM-Plex-Serif-Bold-Italic',
        'Liberation-Serif-Italic',
        'Liberation-Serif-Bold-Italic',
        'NimbusRoman-Italic', # Times
        'NimbusRoman-BoldItalic',
        'Noto-Serif-Italic',
        'Noto-Serif-Bold-Italic',
        'P052-Italic', # Palatino
        'P052-Bold-Italic',
        'URWBookman-LightItalic',
        'URWBookman-DemiItalic',
    ]
    isItalic = random.choices([True, False], [1, 3])[0]
    if isItalic:
        font = random.choice(italicFonts)
    else:
        font = random.choice(fonts)

    alpha = 1.0
    if size >= 48:
        alpha = min(1.0, random.randrange(2, 20)/10)

    def randomColorStr(lighter=None):
        textColorArr = [random.randrange(360)] + [random.randrange(0,256) for _ in range(2)]
        if lighter is not None:
            textColorArr[2] = textColorArr[2] // 2 + 128 if lighter else 0
        return f'hsla({textColorArr[0]},{textColorArr[1]},{textColorArr[2]},{alpha})'

    white = f'rgba(255,255,255,{alpha})'
    black = f'rgba(0,0,0,{alpha})'

    # choose style
    styleNum = random.randrange(100)
    textColorStr = None
    underColorStr = None
    borderColorStr = None
    # if styleNum < 1:
    #     textColorStr = white
    #     underColorStr = black
    # elif styleNum < 2:
    #     textColorStr = black
    #     underColorStr = white
    # elif styleNum < 3:
    #     textColorStr = 'white'
    #     underColorStr = randomColorStr()
    # elif styleNum < 4:
    #     textColorStr = 'black'
    #     underColorStr = randomColorStr()
    # elif styleNum < 5:
    #     textColorStr = randomColorStr()
    #     underColorStr = 'white'
    # elif styleNum < 6:
    #     textColorStr = randomColorStr()
    #     underColorStr = 'black'
    if styleNum < 8:
        textColorStr = randomColorStr()
        underColorStr = randomColorStr()
    # elif styleNum < 9:
    #     textColorStr = white
    # elif styleNum < 14:
    #     textColorStr = 'white'
    #     borderColorStr = 'black'
    # elif styleNum < 15:
    #     textColorStr = randomColorStr(True)
    #     borderColorStr = 'black'
    # elif styleNum < 16:
    #     textColorStr = 'white'
    #     borderColorStr = randomColorStr(False)
    else:
        textColorStr = randomColorStr()

    flags = ['-fill', textColorStr, '-pointsize', str(size), '-font', font, '-gravity', 'center', '-kerning', str(kerning)]
    annotateFlags = ['-annotate', f'{angle}x{angle}{offsetX:+d}{offsetY:+d}', randomString]
    if borderColorStr is not None:
        strokeWidth = random.randrange(1, 5) * 2
        flags += ['-strokewidth', str(strokeWidth), '-stroke', borderColorStr]
        flags += annotateFlags
        # annotate again with no border to remove the border that grows inwards
        flags += ['-stroke', 'none']
        flags += annotateFlags
    else:
        if underColorStr is not None:
            flags += ['-undercolor', underColorStr]
        flags += annotateFlags

    return flags

def processInFile(inFile):
    inFilePath = os.path.join(inDir, inFile)
    inX, inY = Image.open(inFilePath).size
    baseName = os.path.splitext(inFile)[0]
    lowPngCompressionFlags = ['-quality', '0'] # use worse compression for temporary files to go faster

    name, flags = resizeOptions(inX, inY)
    # Insert text before or after resize/blur/sharpen
    textMode = None
    if TEXT:
        if inX >= 512:
            textMode = random.choices([None, 'clean', 'degraded'], [1, 1, 3])[0]
        elif inX >= 256:
            textMode = random.choices([None, 'clean', 'degraded'], [4, 1, 3])[0]

    if textMode:
        if textMode == 'clean':
            name = f'{name}-text'
            minSize = 24
        else:
            name = f'text-{name}'
            minSize = 36
        textFlags = textOptions(inX, inY, minSize)

    # there is no lossless option. high quality compressed is close enough
    # we need both jpg and asp. imagemagick's jpg leans towards mosquito noise. ffmpeg's asp encoder leans toward blocking.
    compression = random.choices(['jpg', 'asp', 'h264', 'vp9', 'h265'], [10, 15, 40, 15, 20])[0]
    if compression == 'jpg':
        quality = random.randrange(70 if HQ_COMP else 60, 90)
    elif compression == 'asp':
        quality = random.randrange(1, 7 if HQ_COMP else 8)
    elif compression == 'h264':
        quality = random.randrange(16, 23 if HQ_COMP else 25)
    elif compression == 'vp9':
        quality = random.randrange(30, 45 if HQ_COMP else 50)
    elif compression == 'h265':
        quality = random.randrange(19, 26 if HQ_COMP else 28)
    else:
        assert False
    quality = str(quality)
    name += f'-{compression}-{quality}'

    outBaseName = baseName + '-' + name
    outFilePath = os.path.join(outDirLR, outBaseName+'.png')
    outLlFilePath = os.path.join(outDirLR, outBaseName+'.ll.png')
    outHrFilePath = os.path.join(outDirHR, outBaseName+'.png')

    print(outBaseName)

    if textMode == 'degraded':
        subprocess.run(['convert', inFilePath] + textFlags + [outHrFilePath], check=True, capture_output=True)
        subprocess.run(['convert', outHrFilePath] + flags + lowPngCompressionFlags + [outLlFilePath], check=True, capture_output=True)
    elif textMode == 'clean':
        textLayerFilePath = os.path.join(outDirHR, outBaseName+'-textlayer.png')
        subprocess.run(['convert', '-size', f'{inX}x{inY}', 'xc:transparent'] + textFlags + lowPngCompressionFlags + [textLayerFilePath], check=True, capture_output=True)
        subprocess.run(['convert', inFilePath, textLayerFilePath, '-composite', outHrFilePath], check=True, capture_output=True)
        textResizeFilter = random.choice(['Box', 'Triangle', 'Catrom', 'Lanczos', 'Spline'])
        subprocess.run(['convert', inFilePath] + flags + ['(', textLayerFilePath, '-filter', textResizeFilter, '-resize', f'{inX//FACTOR}x{inY//FACTOR}!', ')', '-composite'] + lowPngCompressionFlags + [outLlFilePath], check=True, capture_output=True)
        os.remove(textLayerFilePath)
    else:
        subprocess.run(['convert', inFilePath] + flags + lowPngCompressionFlags + [outLlFilePath], check=True, capture_output=True)
        os.symlink(os.path.join('..', inDirBaseName, inFile), outHrFilePath)

    if compression == 'jpg':
        outTmpFilePath = outFilePath+'.jpg'
        subprocess.run(['convert', outLlFilePath, '-quality', quality, outTmpFilePath], check=True, capture_output=True)
        subprocess.run(['convert', outTmpFilePath, outFilePath], check=True, capture_output=True)
        os.remove(outTmpFilePath)
    elif compression == 'asp':
        outTmpFilePath = outFilePath+'.avi'
        # full_chroma_int+accurate_rnd is needed for good quality chroma subsampling, especially from 8 bit yuv420p. ffmpeg's default is quite poor.
        subprocess.run(['ffmpeg', '-i', outLlFilePath, '-q', quality, '-pix_fmt', 'yuv420p', '-sws_flags', '+full_chroma_int+accurate_rnd', outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24', '-sws_flags', '+full_chroma_int+accurate_rnd', outFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        os.remove(outTmpFilePath)
    elif compression == 'h264':
        outTmpFilePath = outFilePath+'.264'
        x264tune = random.choice([[], ['-tune', 'film'], ['-tune', 'grain']])
        # full_chroma_int+accurate_rnd is needed for good quality chroma subsampling, especially from 8 bit yuv420p. ffmpeg's default is quite poor.
        subprocess.run(['ffmpeg', '-i', outLlFilePath, '-crf', quality] + x264tune + [ '-pix_fmt', 'yuv420p', '-sws_flags', '+full_chroma_int+accurate_rnd', outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24', '-sws_flags', '+full_chroma_int+accurate_rnd', outFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        os.remove(outTmpFilePath)
    elif compression == 'h265':
        outTmpFilePath = outFilePath+'.265'
        # x265 uses ~40% psy-rd strength for I frames. See source/encoder/rdcost.h. To get more typical distortions (microbanding) seen on P/B frames, increase psyRd from default 2 to 5.
        x265params = 'psy-rd=5'
        if random.randrange(2) == 0:
            # Disabling sao is a popular setting. It also generates stronger artifacts
            x265params += ',sao=0'
        subprocess.run(['ffmpeg', '-i', outLlFilePath, '-crf', quality, '-x265-params', x265params, '-pix_fmt', 'yuv420p10le', '-sws_flags', '+full_chroma_int+accurate_rnd', outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24', '-sws_flags', '+full_chroma_int+accurate_rnd', outFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        os.remove(outTmpFilePath)
    elif compression == 'vp9':
        outTmpFilePath = outFilePath+'.webm'
        subprocess.run(['ffmpeg', '-i', outLlFilePath, '-crf', quality, '-pix_fmt', 'yuv420p', '-sws_flags', '+full_chroma_int+accurate_rnd', '-c:v', 'libvpx-vp9', outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24', '-sws_flags', '+full_chroma_int+accurate_rnd', outFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        os.remove(outTmpFilePath)
    else:
        assert False
    os.remove(outLlFilePath)

inFiles = os.listdir(inDir)

with multiprocessing.pool.ThreadPool(24) as threadPool:
#    threadPool.map(processInFile, inFiles)
    for _ in tqdm.tqdm(threadPool.imap_unordered(processInFile, inFiles), total=len(inFiles)):
        pass
