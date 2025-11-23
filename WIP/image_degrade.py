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
import cv2

FACTOR = 2
FACTOR_LOG = 1
HQ_SCALE = False
TEXT = True
NOISE = True
NOISE_FRAMES = 10
NOISE_REPEAT = 3
# Mix 2 random degradations together by merging with a mask. This has 2 purposes
# 1. to simulate adaptive deinterlacing affecting only part of the picture
# 2. to train model to remove out of focus blur
MIXED_DEGRADATION = True

characters = string.ascii_letters + string.digits + string.punctuation
# remove symbols that need escaping
characters = characters.replace('%', '').replace('@', '').replace('\\', '')

rgbToYuv = [
    [ 0.2126,  0.7152,  0.0722 ],
    [ -0.1146, -0.3854, 0.5 ],
    [ 0.5,     -0.4542, -0.0458 ],
]
yuvToRgb = np.linalg.inv(rgbToYuv)

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

        if sharpeningMode:
            # Sharpening in linear light produces extremely nasty dark halos but some real sources are actually like that.
            # Sharpening in gamma light produces more bright halo and less dark halo
            # Surprisingly, using linear light sharpening all the time when using linear light scaling (45% of the time) doesn't derail the model.
            sharpenInGamma = True
            if useLinearLight and sharpenInGamma:
                flags += gammaLightFlags

            if sharpeningMode == 'unsharp':
                radius = random.choice([1,2,3])
                sigma = random.randrange(1, 60 - radius * 10) / 10
                name += f'unsharp{radius}x{sigma}-'
                sharpeningFlags = ['-sharpen', f'{radius}x{sigma}']
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
                sharpeningFlags = ['-morphology', 'Convolve', f'{hTaps}x{vTaps}: {kernelStr}']

            if True:
                # Use an edge mask to merge the sharpened copy with the original. This trains the model to fix oversharpened edges without softening the whole image.
                # While softening the whole image is objectively accurate, it looks uninspiring.
                flags += ['(', '+clone'] + sharpeningFlags + ['(', '+clone', '-colorspace', 'Gray', '-canny', '0x1+10%+30%', '-morphology', 'Dilate', 'Disk', '-clamp', '-blur', '0x3', ')', '-compose', 'CopyOpacity', '-composite', ')', '-compose', 'Over', '-composite']
            else:
                flags += sharpeningFlags

            if useLinearLight and sharpenInGamma:
                flags += linearLightFlags

        # Deinterlacing is assumed to be in gamma light.
        if isInterlaced:
            if useLinearLight:
                flags += gammaLightFlags
            name += f'{deintFilter}-'
            flags += ['-filter', deintFilter, '-resize', f'{downX}x{downY*2}!']
            if useLinearLight:
                flags += linearLightFlags

    name += f'{finalFilter}'
    flags += ['-filter', finalFilter, '-resize', f'{outX}x{outY}!']

    if useLinearLight:
        flags += gammaLightFlags # output gamma light

    return name, flags

def mixFlags(inX, inY, flags1, flags2):
    #-alpha off '(' -clone -1 -resize 256x256 -resize 512x512 ')' '(' -clone -2 -flip '(' -size 512x512 xc:black -fill white -draw 'translate 200,200 circle 0,0 100,0' -alpha off ')' -compose CopyOpacity -composite ')' -delete -3 -compose Over -composite
    outX = inX // FACTOR
    outY = inY // FACTOR
    circleX = random.randrange(0, outX)
    circleY = random.randrange(0, outY)
    circleRadius = random.randrange((outX + outY) // 8, (outX + outY) // 4)
    mergeMaskFlags = ['(', '-size', f'{outX}x{outY}', 'xc:black', '-fill', 'white', '-draw', f'translate {circleX},{circleY} circle 0,0 {circleRadius},0', '-blur', '0x2', '-alpha', 'off', ')']
    return ['-alpha', 'off', '(', '-clone', '-1'] + flags1 + [')', '(', '-clone', '-2'] + flags2 + mergeMaskFlags + ['-compose', 'CopyOpacity', '-composite', ')', '-delete', '-3', '-compose', 'Over', '-composite']

def stripLinearLightFlags(flags):
    cleanedFlags = []
    i = 0
    while i < len(flags):
        if flags[i] == '-colorspace':
            i += 2 # skip next
        else:
            cleanedFlags += [flags[i]]
            i+= 1
    return cleanedFlags

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
    # elif styleNum < 10:
    #     textColorStr = 'white'
    #     borderColorStr = 'black'
    elif styleNum < 15:
        textColorStr = randomColorStr(True)
        borderColorStr = 'black'
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

    if MIXED_DEGRADATION:
        if True: # The effect is mild, so it can be on all the time
            name2, flags2 = resizeOptions(inX, inY)
            flags = mixFlags(inX, inY, flags, flags2)
            name = name + '-MIX-' + name2

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

    noiseType = None
    noiseRelativeStrength = 0
    noiseAtten = 1
    noiseChStr = [1, 1, 1]
    noiseSaturation = 100
    noiseDegrade = False
    noiseDownFilter = None
    if NOISE:
        noiseType = random.choices([None, 'Gaussian', 'Poisson', 'Laplacian'], [38, 1, 1, 0])[0]
        if noiseType is not None:
            noiseRelativeStrength = random.random() # 0 = weak, 1 = strong
            if noiseType == 'Gaussian':
                noiseAtten = int(3 + noiseRelativeStrength * 17) / 10 # higher = stronger
            elif noiseType == 'Laplacian':
                noiseAtten = int(5 + noiseRelativeStrength * 25) / 10 # higher = stronger
            elif noiseType == 'Poisson':
                noiseAtten = int(1 / (1/63 + noiseRelativeStrength * (1/2 - 1/63))) # higher = weaker. More than 64 misbehaves
            noiseChStr= []
            while sum(noiseChStr) <= 0:
                noiseChStr = [max(0.0, random.uniform(-1.0, 1.0)) for _ in range(3)]
            noiseChStrAvg = sum(noiseChStr) / len(noiseChStr)
            noiseChStr = [chStr / noiseChStrAvg for chStr in noiseChStr]
            # Chroma subsampling will soften chroma noise, so allow noise saturation adjustment to go over 100% to compensate
            noiseSaturation = int(random.random() * 150)
            noiseDegrade = True # random.choice([True, False])
            name = f'{name}-n{noiseType}{noiseAtten}x' + ''.join(f'{min(int(chStr*10), 15):x}' for chStr in noiseChStr) + f's{noiseSaturation}'
            if noiseDegrade:
                name += 'd'
            else:
                noiseDownFilter = random.choice(['Mitchell', 'Catrom', 'Lanczos', 'Gaussian', 'Spline', 'Triangle'])
                name += noiseDownFilter

    # The goal of training with anamorphic encoding is to train the model to recognize resized compression artifacts
    anamorphicWidth = inX // FACTOR
    if random.randrange(3) == 0:
        anamorphicWidth = random.randrange(int(inX / FACTOR * 3 / 5), int(inX / FACTOR * 6 / 5)) // 2 * 2
        name += f'-ana{anamorphicWidth}'

    # there is no lossless option. high quality compressed is close enough
    compression = random.choices(['asp', 'mpg', 'h264', 'vp9', 'h265'], [15, 15, 35, 15, 20])[0]
    if compression == 'asp':
        qualityRange = (7, 1)
    elif compression == 'mpg':
        qualityRange = (7, 1)
    elif compression == 'h264':
        qualityRange = (25, 14) if noiseType else (23, 16)
    elif compression == 'vp9':
        qualityRange = (40, 25) if noiseType else (45, 30) # at low quality levels vp9 turns noise into horrific DCT artifacts
    elif compression == 'h265':
        qualityRange = (28, 17) if noiseType else (26, 19)
    else:
        assert False

    normalizedCompQuality = random.random() # 0 = worst, 1 = best
    quality = qualityRange[0] + normalizedCompQuality * (qualityRange[1] - qualityRange[0])

    quality = str(int(quality))
    name += f'-{compression}-{quality}'

    outBaseName = baseName + '-' + name
    outFilePath = os.path.join(outDirLR, outBaseName+'.png')
    outLlFilePath = os.path.join(outDirLR, outBaseName+'.ll.png')
    outHrFilePath = os.path.join(outDirHR, outBaseName+'.png')

    tempFiles = [outLlFilePath]

    #print(outBaseName)

    if textMode == 'degraded':
        subprocess.run(['convert', inFilePath] + textFlags + flags + lowPngCompressionFlags + [outLlFilePath], check=True, capture_output=True)
    elif textMode == 'clean':
        textLayerFilePath = os.path.join(outDirHR, outBaseName+'-textlayer.png')
        tempFiles += [textLayerFilePath]
        subprocess.run(['convert', '-size', f'{inX}x{inY}', 'xc:transparent'] + textFlags + lowPngCompressionFlags + [textLayerFilePath], check=True, capture_output=True)
        textResizeFilter = random.choice(['Box', 'Triangle', 'Catrom', 'Lanczos', 'Spline'])
        subprocess.run(['convert', inFilePath] + flags + ['(', textLayerFilePath, '-filter', textResizeFilter, '-resize', f'{inX//FACTOR}x{inY//FACTOR}!', ')', '-composite'] + lowPngCompressionFlags + [outLlFilePath], check=True, capture_output=True)
    else:
        subprocess.run(['convert', inFilePath] + flags + lowPngCompressionFlags + [outLlFilePath], check=True, capture_output=True)

    ffmpegInputFile = outLlFilePath
    ffmpegOutputFile = outFilePath
    # Noise may be degraded or clean. In real videos especially film, noise can be sharper than content.
    # Even for degraded noise, we degrade the noise separately and add it to the LR and HR at the end. We can't add it to HR first then generate degraded LR because the HR will have more noise than LR.
    if noiseType is not None:
        if noiseDegrade:
            # scaling in linear light changes the noise brightness too much. Always do noise degradation in gamma light.
            # Don't use mixed degradation for noise. With mixed noise degradation it is impossible to match HR and LR noise amount.
            #noiseDegradeFlags = stripLinearLightFlags(flags2 if MIXED_DEGRADATION else flags)
            _, noiseDegradeFlags = resizeOptions(inX, inY)
            noiseDegradeFlags = stripLinearLightFlags(noiseDegradeFlags)
        else:
            noiseDegradeFlags = ['-filter', noiseDownFilter, '-resize', f'{inX//2}x{inY//2}']
        if random.randrange(2) == 0:
            noiseDegradeFlags += ['-blur', f'0x{random.randint(1, 10)/10}']

        noiseGenFlags = ['-duplicate', '1', # stack: HR1, HR2
                        '-attenuate', str(noiseAtten), '+noise', noiseType, # add noise to HR2
                        '-compose', 'Mathematics', '-define', 'compose:args=0,1,-1,0.5', '-composite', # noise = HR2 - HR1 + 0.5
                        '-color-matrix', f'6x3: {noiseChStr[0]} 0 0 0 0 {0.5*(1-noiseChStr[0])}  0 {noiseChStr[1]} 0 0 0 {0.5*(1-noiseChStr[1])}  0 0 {noiseChStr[2]} 0 0 {0.5*(1-noiseChStr[2])}',
                        '-modulate', f'100,{noiseSaturation}']

        allNoiseGenFlags = []
        allLrFramesFlags = []
        # Simulate both I frame and P frame compression
        noiseFrames = NOISE_FRAMES # random.choice([1, NOISE_FRAMES])
        noiseRepeat = NOISE_REPEAT if noiseFrames > 1 else 1
        for i in range(noiseFrames):
            noiseFrameFilePath = os.path.join(outDirHR, outBaseName+f'-noiselayer{i}.png')
            lrFrameFilePath = os.path.join(outDirLR, outBaseName+f'-ll{i}.png')
            tempFiles += [noiseFrameFilePath, lrFrameFilePath]
            allNoiseGenFlags += ['(', '+clone'] + noiseGenFlags + ['-write', noiseFrameFilePath, '+delete', ')']
            allLrFramesFlags += ['(', '+clone', '(', noiseFrameFilePath ] + noiseDegradeFlags + [ ')' , '-compose', 'Mathematics', '-define', 'compose:args=0,1,1,-0.5', '-composite', '-write', lrFrameFilePath, '+delete', ')']

        subprocess.run(['convert', inFilePath, '-depth', '16'] + lowPngCompressionFlags + allNoiseGenFlags + ['/dev/null'], check=True, capture_output=True)
        subprocess.run(['convert', outLlFilePath] + lowPngCompressionFlags + allLrFramesFlags + ['/dev/null'], check=True, capture_output=True)
        for rep in range(1, noiseRepeat):
            for i in range(noiseFrames):
                srcFile = outBaseName+f'-ll{i}.png'
                dstFile = os.path.join(outDirLR, outBaseName+f'-ll{i + rep * noiseFrames}.png')
                os.symlink(srcFile, dstFile)
                tempFiles += [dstFile]
        for i in range(noiseFrames * noiseRepeat):
            tempFiles += [os.path.join(outDirLR, outBaseName+f'-{i+1}.png')] # output starts at 1
        ffmpegInputFile = os.path.join(outDirLR, outBaseName+'-ll%d.png')
        ffmpegOutputFile = os.path.join(outDirLR, outBaseName+'-%d.png')

    # full_chroma_int+accurate_rnd is needed for good quality chroma subsampling, especially from 8 bit yuv420p. ffmpeg's default is quite poor.
    ffHqChromaFlags = ['-sws_flags', '+full_chroma_int+accurate_rnd']
    # note: anamorphicWidth may be equal to inX//FACTOR, which makes these flags noop
    # Ideally scaling to the anamorphic resolution should be done as a part of the normal degradation but the interaction with noise and text is too complex, so we do it at the end.
    ffPreCompressionResizeFlags = ['-s', f'{anamorphicWidth}x{inY//FACTOR}']
    ffPostCompressionResizeFlags = ['-s', f'{inX//FACTOR}x{inY//FACTOR}']

    if compression == 'asp':
        outTmpFilePath = outFilePath+'.avi'
        subprocess.run(['ffmpeg', '-i', ffmpegInputFile, '-q', quality, '-g', '1000', '-keyint_min', '1000', '-pix_fmt', 'yuv420p'] + ffPreCompressionResizeFlags + ffHqChromaFlags + [outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24'] + ffPostCompressionResizeFlags + ffHqChromaFlags + [ffmpegOutputFile], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        tempFiles += [outTmpFilePath]
    elif compression == 'mpg':
        outTmpFilePath = outFilePath+'.mpg'
        subprocess.run(['ffmpeg', '-i', ffmpegInputFile, '-q', quality, '-g', '1000', '-keyint_min', '1000', '-pix_fmt', 'yuv420p', '-c:v', 'mpeg2video'] + ffPreCompressionResizeFlags + ffHqChromaFlags + [outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        overwriteFlags = ['-update', '1'] if '%d' not in ffmpegOutputFile else []
        subprocess.run(['ffmpeg', '-c:v', 'mpeg2video', '-i', outTmpFilePath, '-pix_fmt', 'rgb24'] + ffPostCompressionResizeFlags + ffHqChromaFlags + overwriteFlags + [ffmpegOutputFile], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        tempFiles += [outTmpFilePath]
    elif compression == 'h264':
        outTmpFilePath = outFilePath+'.mkv'
        x264flags = ['-g', '1000', '-keyint_min', '1000']
        x264flags += random.choice([[], ['-tune', 'film'], ['-tune', 'grain'], ['-tune', 'grain', '-deblock', '-3:-3']])
        x264flags += ['-preset', random.choice(['fast', 'medium', 'slow', 'slower'])]
        x264flags += random.choice([[], ['-mbtree', '0']])
        x264flags += ['-aq-strength', str(random.randint(7, 13) / 10)]
        subprocess.run(['ffmpeg', '-i', ffmpegInputFile, '-c:v', 'libx264', '-crf', quality] + x264flags + [ '-pix_fmt', 'yuv420p'] + ffPreCompressionResizeFlags + ffHqChromaFlags + [outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24'] + ffPostCompressionResizeFlags + ffHqChromaFlags + [ffmpegOutputFile], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        tempFiles += [outTmpFilePath]
    elif compression == 'h265':
        outTmpFilePath = outFilePath+'.mkv'
        x265Preset = random.choice(['medium', 'slow'])
        x265params = f'keyint=1000:min-keyint=1000:preset={x265Preset}'
        if random.randrange(2) == 0:
            # Disabling sao is a popular setting. It also generates stronger artifacts
            x265params += ':sao=0'
        subprocess.run(['ffmpeg', '-i', ffmpegInputFile, '-c:v', 'libx265', '-crf', quality, '-x265-params', x265params, '-pix_fmt', 'yuv420p10le'] + ffPreCompressionResizeFlags + ffHqChromaFlags + [outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24'] + ffPostCompressionResizeFlags + ffHqChromaFlags + [ffmpegOutputFile], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        tempFiles += [outTmpFilePath]
    elif compression == 'vp9':
        outTmpFilePath = outFilePath+'.webm'
        subprocess.run(['ffmpeg', '-i', ffmpegInputFile, '-crf', quality, '-g', '1000', '-keyint_min', '1000','-pix_fmt', 'yuv420p'] + ffPreCompressionResizeFlags + ffHqChromaFlags + ['-c:v', 'libvpx-vp9', outTmpFilePath], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        subprocess.run(['ffmpeg', '-i', outTmpFilePath, '-pix_fmt', 'rgb24'] + ffPostCompressionResizeFlags + ffHqChromaFlags + [ffmpegOutputFile], check=True, capture_output=True, stdin=subprocess.DEVNULL)
        tempFiles += [outTmpFilePath]
    else:
        assert False

    if textMode == 'degraded':
        subprocess.run(['convert', inFilePath] + textFlags + [outHrFilePath], check=True, capture_output=True)
    elif textMode == 'clean':
        subprocess.run(['convert', inFilePath, textLayerFilePath, '-composite'] + [outHrFilePath], check=True, capture_output=True)
    else:
        subprocess.run(['cp', inFilePath, outHrFilePath])

    if noiseType is not None:
        noiseFrameToUse = random.randrange(0, noiseFrames * noiseRepeat)
        noiseLayerFilePath = os.path.join(outDirHR, outBaseName+f'-noiselayer{noiseFrameToUse%noiseFrames}.png')
        lrFrameFilePath = os.path.join(outDirLR, outBaseName+f'-{noiseFrameToUse+1}.png')
        os.rename(lrFrameFilePath, outFilePath)
        tempFiles.remove(lrFrameFilePath)
        # The amount of noise that makes it to the end is too hard to predict, so we have to measure it after compressing the image.
        # compressedLrPix = np.array(Image.open(outFilePath).convert('RGB').getdata()).T
        # noiseLrPix = np.array(Image.open(noiseLayerFilePath).resize((inX//2, inY//2)).convert('RGB').getdata()).T
        # compressedLrYuv = rgbToYuv @ compressedLrPix
        # noiseLrYuv = rgbToYuv @ noiseLrPix
        noisePix = np.float32(cv2.imread(noiseLayerFilePath, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)) * 255 / 65535
        #noisePix = cv2.GaussianBlur(noisePix, (5, 5), 0.5)
        #compressedLrPix = np.float32(cv2.imread(os.path.join(outDirLR, outBaseName+f'-ll9.png'), cv2.IMREAD_COLOR))
        compressedLrPix = np.float32(cv2.imread(outFilePath, cv2.IMREAD_COLOR))

        noiseBlurred = [cv2.GaussianBlur(noisePix, (int(1 + 4 * (2**i)), int(1 + 4 * (2**i))), 2**i) for i in range(-1 + FACTOR_LOG, 3 + FACTOR_LOG)]
        compressedLrBlurred = [cv2.GaussianBlur(compressedLrPix, (int(1 + 4 * (2**i)), int(1 + 4 * (2**i))), 2**i) for i in range(-1, 3)]
        noiseDog = [noisePix - noiseBlurred[0]]
        noiseDogLr = []
        compressedLrDog = [compressedLrPix - compressedLrBlurred[0]]
        for i in range(len(noiseBlurred) - 1):
            noiseDog += [noiseBlurred[i] - noiseBlurred[i+1]]
            compressedLrDog += [compressedLrBlurred[i] - compressedLrBlurred[i+1]]
        for i in range(len(noiseDog)):
            noiseDogLr += [cv2.resize(noiseDog[i], (inX//FACTOR, inY//FACTOR))]

        noiseMatchedByDog = noisePix * 0 # same shape as noisePix but all 0
        yuvScaleDog = []
        for dogLevel in reversed(range(len(noiseDog))):
            noiseLrLevelYuv = cv2.cvtColor(noiseDogLr[dogLevel], cv2.COLOR_BGR2YUV)
            compressedLrLevelYuv = cv2.cvtColor(compressedLrDog[dogLevel], cv2.COLOR_BGR2YUV)
            yuvScaleDogLevel = []
            for channel in range(3):
                noiseLrLevelChan = noiseLrLevelYuv[:,:,channel]
                compressedLrLevelChan = compressedLrLevelYuv[:,:,channel]
                covarianceMatrix = np.cov([noiseLrLevelChan.reshape(inX//FACTOR * inY//FACTOR), compressedLrLevelChan.reshape(inX//FACTOR * inY//FACTOR)])
                yuvScaleDogLevel.append(max(0.0, covarianceMatrix[0][1] / covarianceMatrix[0][0]))
                # debug
                # if channel == 0:
                #     for i in range(len(noiseLrLevelChan)): cv2.imwrite(f'{noiseLayerFilePath}-dog{dogLevel}.png', np.uint8(noiseLrLevelChan)+128)
                #     for i in range(len(compressedLrLevelChan)): cv2.imwrite(f'{outFilePath}-dog{dogLevel}.png', np.uint8(compressedLrLevelChan)+128)
            if yuvScaleDog:
                yuvScaleDogLevel[0] = max(yuvScaleDogLevel[0], yuvScaleDog[-1][0] * 0.8) # limit spectrum decay rate
            noiseMatchedByDog += noiseDog[dogLevel] * min(3.0, yuvScaleDogLevel[0] * 1.5)
            yuvScaleDog += [yuvScaleDogLevel]
        # cv2.imwrite(f'{noiseLayerFilePath}-dogmatch.png', np.uint8(noiseMatchedByDog)+128)

        hrPix = np.float32(cv2.imread(outHrFilePath, cv2.IMREAD_COLOR))
        hrPix += noiseMatchedByDog
        cv2.imwrite(outHrFilePath, np.uint8(np.clip(hrPix, 0, 255)))

        # old version, doesn't do DoG decomposition

        # noiseLrPix = cv2.resize(noisePix, (inX//FACTOR, inY//FACTOR))
        # noiseLrYuv = cv2.cvtColor(noiseLrPix, cv2.COLOR_BGR2YUV)
        # compressedLrYuv = cv2.cvtColor(compressedLrPix, cv2.COLOR_BGR2YUV)
        # yuvScale = []
        # for channel in range(3):
        #     compressedLrChan = compressedLrYuv[:,:,channel]
        #     noiseLrChan = noiseLrYuv[:,:,channel]
        #     covarianceMatrix = np.cov([noiseLrChan.reshape(inX//FACTOR * inY//FACTOR), compressedLrChan.reshape(inX//FACTOR * inY//FACTOR)])
        #     yuvScale.append(max(0.0, covarianceMatrix[0][1] / covarianceMatrix[0][0]))

        # # The color matrix must not scale Y. Y is centred at 50% while UV are centred at 0.
        # hrNoiseScale = min(3.0, yuvScale[0] * 1.3)
        # if hrNoiseScale > 0:
        #     yuvScale[1] = min(1.0, yuvScale[1] * 1.6) / hrNoiseScale
        #     yuvScale[2] = min(1.0, yuvScale[2] * 1.6) / hrNoiseScale
        # yuvScale[0] = 1.0
        # yuvScaleMatrix = np.identity(3) * yuvScale
        # colorMatrix = yuvToRgb @ yuvScaleMatrix @ rgbToYuv
        #
        # # covarianceMatrix = np.cov([noiseLrPix, compressedLrPix])
        # # # subjective fudge factor to compensate for differences in scaling method, blur, and compression roughening noise.
        # # hrNoiseScale = max(0.0, min(1.0, 1.2 * covarianceMatrix[0][1] / covarianceMatrix[0][0]))
        # print(hrNoiseScale, yuvScale)
        # hrNoiseFlags = [ '(', noiseLayerFilePath,
        #                 '-blur', '0x0.5',
        #                # '-color-matrix', ' '.join(' '.join(f'{coef:.04f}' for coef in row) for row in colorMatrix),
        #                 ')',
        #                 '-compose', 'Mathematics', '-define', f'compose:args=0,{hrNoiseScale},1,-{hrNoiseScale/2}', '-composite']

        # if hrNoiseFlags:
        #     subprocess.run(['convert', inFilePath] + hrNoiseFlags + [outHrFilePath], check=True, capture_output=True)
        # else:
        #     os.symlink(os.path.join('..', inDirBaseName, inFile), outHrFilePath)


    for filePath in tempFiles:
        os.remove(filePath)

inFiles = os.listdir(inDir)

# try:
#     for inFile in inFiles:
#         processInFile(inFile)
# except Exception as e:
#     pdb.post_mortem(e)
#     exit(1)

with multiprocessing.pool.ThreadPool(24) as threadPool:
#    threadPool.map(processInFile, inFiles)
    for _ in tqdm.tqdm(threadPool.imap_unordered(processInFile, inFiles), total=len(inFiles)):
        pass
