#!/usr/bin/python3
from PIL import Image
import sys
import os
import pdb
import random
import subprocess
import multiprocessing.pool

FACTOR = 2
MOD = 4 # make output sizes mod 4
MINDIM = 256 # minimum output dimension, smaller images are dropped
factorName = 'half' if FACTOR == 2 else 'quarter'

inDir = sys.argv[1]
parentDir, inDirBaseName = os.path.split(inDir.rstrip('/'))
try:
    outDirLR = sys.argv[2]
except IndexError:
    outDirLR = os.path.join(parentDir, f'{inDirBaseName}_{factorName}')

try:
    os.mkdir(outDirLR)
except FileExistsError:
    pass

def resizeOptions(inX, inY):
    flags = []

    hrMod = MOD * FACTOR
    if inX % hrMod or inY % hrMod:
        inX -= inX % hrMod
        inY -= inY % hrMod
        flags += ['-crop', f'{inX}x{inY}+0+0!']

    linearLight = False

    if linearLight:
        flags += ['-colorspace', 'RGB']

    downFilter = 'Triangle' # random.choice(['Mitchell', 'Triangle', 'Spline', 'Gaussian', 'Catrom', 'Robidoux', 'Box', 'Quadratic'])
    downX = inX // FACTOR
    downY = inY // FACTOR

    filter = ''
    name = f'{factorName}{downFilter}'
    flags += ['-filter', downFilter, '-resize', f'{downX}x{downY}!']

    if linearLight:
        flags += ['-colorspace', 'sRGB']
        name += 'l'

    return name, flags

def processInFile(inFile):
    inFilePath = os.path.join(inDir, inFile)
    inX, inY = Image.open(inFilePath).size
    if inX < MINDIM * FACTOR or inY < MINDIM * FACTOR:
        return

    baseName = os.path.splitext(inFile)[0]
    name, flags = resizeOptions(inX, inY)
    outBaseName = baseName + '-' + name
    print(outBaseName)
    outFilePath = os.path.join(outDirLR, outBaseName+'.png')
    subprocess.run(['convert', inFilePath] + flags + [outFilePath], check=True, capture_output=True)

inFiles = sorted(os.listdir(inDir))#[::50]
with multiprocessing.pool.ThreadPool(20) as threadPool:
    threadPool.map(processInFile, inFiles)
