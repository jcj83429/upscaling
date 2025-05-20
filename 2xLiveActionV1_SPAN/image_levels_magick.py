#!/usr/bin/python3

# The training datasets tend to have well exposed images without blown-out whites and crushed blacks. The model seems to have difficulty outputting pure white and pure black on-screen graphics. Mess with the levels to create underexposed and overexposed images.

from PIL import Image, ImageStat, ImageEnhance
import sys
import os
import pdb
import random
import subprocess
import multiprocessing.pool

inDir = sys.argv[1]
parentDir, inDirBaseName = os.path.split(inDir)
outDir = os.path.join(parentDir, inDirBaseName+'_levels')

def processInFile(inFile):
    inFilePath = os.path.join(inDir, inFile)

    brightnessFactor = random.uniform(0.75, 1.25)
    contrastFactor = random.uniform(0.75, 1.25)
    saturationFactor = random.uniform(0.75, 1.25)
    wbChannels = random.sample(['R', 'G', 'B'], 2)
    wbChannelScale = {
        wbChannels[0] : random.uniform(0.85, 1.0),
        wbChannels[1] : random.uniform(0.85, 1.0),
    }

    # apply in random order
    effects = ['brightness', 'contrast', 'saturation', 'wb']
    random.shuffle(effects)
    name = os.path.splitext(inFile)[0]
    flags = []
    for effect in effects:
        if effect == 'brightness':
            name += f'_bri{brightnessFactor:.02f}'
            flags += ['-brightness-contrast', f'{int(brightnessFactor * 100 - 100)}x0']
        elif effect == 'contrast':
            name += f'_con{contrastFactor:.02f}'
            flags += ['-brightness-contrast', f'0x{int(contrastFactor * 100 - 100)}']
        elif effect == 'saturation':
            name += f'_sat{saturationFactor:.02f}'
            flags += ['-modulate', f'100,{int(saturationFactor * 100)},100']
        elif effect == 'wb':
            name += f'_{wbChannels[0]}{int(wbChannelScale[wbChannels[0]]*100)}{wbChannels[1]}{int(wbChannelScale[wbChannels[1]]*100)}'
            flags += ['-channel', wbChannels[0], '-evaluate', 'multiply', f'{wbChannelScale[wbChannels[0]]:.02f}',
                      '-channel', wbChannels[1], '-evaluate', 'multiply', f'{wbChannelScale[wbChannels[1]]:.02f}']
    outName = name + '.png'
    print(outName, flags)
    outFilePath = os.path.join(outDir, outName)
    subprocess.run(['convert', inFilePath] + flags + [outFilePath], check=True, capture_output=True)

inFiles = sorted(os.listdir(inDir))
with multiprocessing.pool.ThreadPool(24) as threadPool:
    threadPool.map(processInFile, inFiles)
