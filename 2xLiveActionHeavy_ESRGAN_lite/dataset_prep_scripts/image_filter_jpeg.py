#!/usr/bin/python3
from PIL import Image
import sys
import os
import pdb
import random
import subprocess
import multiprocessing.pool
import re

pathToGod = '/home/archuser/GOD/god'

inDir = sys.argv[1]
parentDir, inDirBaseName = os.path.split(inDir.rstrip('/'))
try:
    outDir = sys.argv[2]
except IndexError:
    outDir = os.path.join(parentDir, f'{inDirBaseName}_filterjpeg')

os.makedirs(outDir, exist_ok=True)

cleanDir = os.path.join(outDir, 'clean')
jpegDir =  os.path.join(outDir, 'jpeg')
otherDir = os.path.join(outDir, 'other')
os.makedirs(cleanDir, exist_ok=True)
os.makedirs(jpegDir, exist_ok=True)
os.makedirs(otherDir, exist_ok=True)

def processInFile(inFile):
    inFilePath = os.path.join(inDir, inFile)
    output = subprocess.run([pathToGod, inFilePath, '128'], check=True, capture_output=True)
    gridsFound = int(re.search(b'JPEG grids found: (\\d+)', output.stdout).group(1))
    if gridsFound == 0:
        targetDir = cleanDir
    elif gridsFound == 1:
        targetDir = jpegDir
    else:
        print(inFile, gridsFound)
        targetDir = otherDir
    outFilePath = os.path.join(targetDir, inFile)
    subprocess.run(['cp', inFilePath, outFilePath], check=True)

inFiles = sorted(os.listdir(inDir))

# try:
#     for inFile in inFiles:
#         processInFile(inFile)
# except Exception as e:
#     pdb.post_mortem(e)
#     exit(1)

inFiles = sorted(os.listdir(inDir))
with multiprocessing.pool.ThreadPool(6) as threadPool:
    threadPool.map(processInFile, inFiles)
