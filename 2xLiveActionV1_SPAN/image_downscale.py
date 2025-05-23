#!/usr/bin/python3
from PIL import Image
import sys
import os
import pdb
import random
import subprocess
import multiprocessing.pool

FACTOR = 2
factorName = 'half' if FACTOR == 2 else 'quarter'

inDir = sys.argv[1]
parentDir, inDirBaseName = os.path.split(inDir)
outDirLR = os.path.join(parentDir, inDirBaseName+f'_{factorName}')

def resizeOptions(n, inX, inY):
    options = {}
    for _ in range(n):
        downFilter = 'Spline'
        downX = inX // FACTOR // 8 * 8 # ensure multiple of 8 so it can be chroma subsampled after downscaling to 1/4
        downY = inY // FACTOR // 8 * 8

        name = f'{factorName}spline'
        flags = ['-filter', downFilter, '-resize', f'{downX}x{downY}!']

        options[name] = flags
    return options

def processInFile(inFile):
    inFilePath = os.path.join(inDir, inFile)
    inX, inY = Image.open(inFilePath).size
    options = resizeOptions(1, inX, inY)
    baseName = os.path.splitext(inFile)[0]
    for name, flags in options.items():
        #outBaseName = baseName + '-' + name
        outBaseName = baseName
        print(outBaseName)
        outFilePath = os.path.join(outDirLR, outBaseName+'.png')
        subprocess.run(['convert', inFilePath] + flags + [outFilePath], check=True, capture_output=True)

inFiles = os.listdir(inDir)
with multiprocessing.pool.ThreadPool(20) as threadPool:
    threadPool.map(processInFile, inFiles)
