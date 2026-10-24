#!/usr/bin/python3
from PIL import Image
import sys
import os
import pdb
import random
import shutil
import subprocess
import multiprocessing.pool
import tqdm

MOD = 4 # make output sizes mod 4
MINDIM = 256 # minimum output dimension, smaller images are dropped

inDir = sys.argv[1]
parentDir, inDirBaseName = os.path.split(inDir.rstrip('/'))
try:
    outDirLR = sys.argv[2]
except IndexError:
    outDirLR = os.path.join(parentDir, f'{inDirBaseName}_tile{MINDIM}')

try:
    os.mkdir(outDirLR)
except FileExistsError:
    pass

def processInFile(inFile):
    inFilePath = os.path.join(inDir, inFile)
    inX, inY = Image.open(inFilePath).size
    if inX < MINDIM or inY < MINDIM:
        return

    xSplits = inX // MINDIM
    ySplits = inY // MINDIM

    if xSplits == 1 and ySplits == 1:
        shutil.copyfile(inFilePath, os.path.join(outDirLR, inFile))
        return

    outX = inX // xSplits
    outX -= outX % MOD
    outY = inY // ySplits
    outY -= outY % MOD

    baseName = os.path.splitext(inFile)[0]
    for xIdx in range(xSplits):
        for yIdx in range(ySplits):
            xOffset = xIdx * outX
            yOffset = yIdx * outY
            outFilePath = os.path.join(outDirLR, f'{baseName}_{xIdx}_{yIdx}.png')
            subprocess.run(['convert', inFilePath, '-crop', f'{outX}x{outY}+{xOffset}+{yOffset}!', 'PNG24:'+outFilePath], check=True, capture_output=True)

inFiles = os.listdir(inDir)
with multiprocessing.pool.ThreadPool(24) as threadPool:
    for _ in tqdm.tqdm(threadPool.imap_unordered(processInFile, inFiles), total=len(inFiles)):
        pass
