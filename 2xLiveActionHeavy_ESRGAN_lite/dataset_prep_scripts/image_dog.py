#!/usr/bin/env python

import cv2
import numpy as np
import pdb
import os
import multiprocessing.pool
import subprocess

NUM_LEVELS = 7
BLOCK_SIZE = 16

inDir = '/mnt/ss/dataset/DF2KCleanHSTri_tile256/'
outDir = '/mnt/ss/dataset/DF2KCleanHSTri_tile256_ss/'

def doDog(inFile):
    inFilePath = os.path.join(inDir, inFile)
    outFilePath = os.path.join(outDir, inFile)
    pix = np.uint16(cv2.imread(inFilePath, 0)) * 255
    #pix = cv2.fastNlMeansDenoising(pix, [5], None, 7, 11, cv2.NORM_L1)
    #pix = cv2.medianBlur(pix, 3) # median filter is edge-preserving
    pix = np.float32(pix) / 255
    #pix = cv2.bilateralFilter(pix, 3, 7, 7)
    inY, inX = pix.shape
    if inY > 2048 or inX > 2048:
        if inY > inX:
            inX = inX * 2048 // inY
            inY = 2048
        else:
            inY = inY * 2048 // inX
            inX = 2048
        pix = cv2.resize(pix, (inX, inY), interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite(f'{outFilePath}-denoise.png', np.uint8(pix))

    blurred = [cv2.GaussianBlur(pix, (int(1 + 4 * (2**i)), int(1 + 4 * (2**i))), 2**i) for i in range(-1, -1 + NUM_LEVELS)]
    dog = [pix - blurred[0]]
    for i in range(len(blurred) - 1):
        dog += [blurred[i] - blurred[i+1]]

    # for dogLevel in range(len(blurred)):
    #     dog[dogLevel] *= pow(2, (NUM_LEVELS - 1 - dogLevel) / 6)

    # for dogLevel in range(len(blurred)):
    #     cv2.imwrite(f'{inFilePath}-dog{dogLevel}.png', np.uint8(dog[dogLevel])+128)

    output = np.uint8(pix) * 0
    stats = [0] * (NUM_LEVELS + 1)

    for y in range(0, inY - BLOCK_SIZE + 1, BLOCK_SIZE):
        for x in range(0, inX - BLOCK_SIZE + 1, BLOCK_SIZE):
            strongestLevel = None
            maxValue = 0
            for dogLevel in range(len(dog)):
                block = dog[dogLevel][ y : y+BLOCK_SIZE , x : x+BLOCK_SIZE ]
                levelValue = np.max(abs(block))
                if levelValue < 3:
                    continue
                if levelValue > 15:
                    strongestLevel = dogLevel
                    break
                if levelValue > maxValue:
                    maxValue = levelValue
                    strongestLevel = dogLevel
            if strongestLevel is not None:
                output[ y : y+BLOCK_SIZE , x : x+BLOCK_SIZE ] = np.ones((BLOCK_SIZE, BLOCK_SIZE)) * 255 * (NUM_LEVELS - strongestLevel) / NUM_LEVELS
                stats[strongestLevel] += 1
            else:
                stats[NUM_LEVELS] += 1

    # average_sharpness = np.average(output)
    avgSharpness = 0
    totalCount = 0
    for i in range(NUM_LEVELS + 1):
        count = stats[NUM_LEVELS - i]
        if i == 0:
            count = count // 2 # give less importance to blank areas
        totalCount += count
        avgSharpness += count * min(i, NUM_LEVELS - 1) # merge the sharpest 2 levels together
    avgSharpness = avgSharpness / totalCount / (NUM_LEVELS - 1) * 100

    print(inFile, avgSharpness)
    outFilePath = os.path.join(outDir, f'{int(avgSharpness)}-' + inFile)
    subprocess.run(['cp', inFilePath, outFilePath], check=True)
    #cv2.imwrite(f'{outFilePath}-dbg.png', output)

files = sorted(os.listdir(inDir))
try:
    os.mkdir(outDir)
except FileExistsError:
    pass

# for f in files:
#     doDog(f)

with multiprocessing.pool.ThreadPool(24) as threadPool:
    threadPool.map(doDog, files)
