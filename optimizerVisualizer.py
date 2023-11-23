#! /usr/bin/python3
# -*- coding: utf8 -*-

import os, time, random, re, glob
from os.path import expanduser
from pathlib import Path
import math
import random
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, utils
from adabelief_pytorch import AdaBelief
import AdaDerivative
from lion_pytorch import Lion
import matplotlib
from matplotlib import pyplot
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation


#python3 ~/Program/OptimizerVisualizer/optimizerVisualizer.py


# Mini Batch
miniBatchSize = 1

#Paths
# Home Path
homePath = expanduser("~")
plotImagePath = homePath + "/Program/OptimizerVisualizer/Plots"


#objectiveFunctionName = 'Rosenbrock'
#objectiveFunctionName = 'SixHumpCamel'
#objectiveFunctionName = 'Sphere'
objectiveFunctionName = 'Custom'


numberOfFrames = 100


pi = torch.tensor(math.pi)
e = torch.tensor(math.e)
deltaZ = 1e-5




def rosenbrockFunction(x, y):
    scaleX = torch.tensor(2)
    scaleY = torch.tensor(2)
    shiftY = torch.tensor(1)
    scaleZ = torch.tensor(2509) # Maxima: 2509.00000000, Minima: 0.00000000
    x = x * scaleX
    y = - y * scaleY + shiftY
    a = 1
    b = 100
    z = a * (x - 1) ** 2 + b * (y - x ** 2) ** 2
    z = z / scaleZ * (1 - deltaZ) + deltaZ
    return z


def sixHumpCamelFunction(x, y):
    scaleX = torch.tensor(2)
    scaleY = torch.tensor(- 1)
    rangeZ = 5.49558687 + 1.03008306 # Maximum: 5.49558687, Minimum: - 1.03008306
    scaleZ = torch.tensor(rangeZ)
    shiftZ = torch.tensor(1.03008306)
    x = x * scaleX
    y = y * scaleY
    term1 = (4 - 2.1 * x ** 2 + (x ** 4) / 3) * x ** 2
    term2 = x * y
    term3 = (- 4 + 4 * y ** 2) * y ** 2
    z = term1 + term2 + term3
    z = (z + shiftZ) / scaleZ * (1 - deltaZ) + deltaZ
    return z


def sphereFunction(x, y):
    shiftY = torch.tensor(- 0.4)
    rangeZ = 2.46000004 # Maxima: 2.46000004, Minima: 0.00000000
    scaleZ = torch.tensor(rangeZ)
    y = - y + shiftY
    z = 0.5 * x ** 2 + y ** 2
    z = z / scaleZ * (1 - deltaZ) + deltaZ
    return z


def customFunction(x, y):
    scaleX1 = torch.tensor(2)
    scaleY1 = torch.tensor(- 1)
    rangeZ1 = 5.49558687 + 1.03008306 # Maximum: 5.49558687, Minimum: - 1.03008306
    scaleZ1 = torch.tensor(rangeZ1)
    shiftZ1 = torch.tensor(1.03008306)
    x1 = x * scaleX1
    y1 = y * scaleY1
    term1 = (4 - 2.1 * x1 ** 2 + (x1 ** 4) / 3) * x1 ** 2
    term2 = x1 * y1
    term3 = (- 4 + 4 * y1 ** 2) * y1 ** 2
    z1= term1 + term2 + term3
    z1 = (z1 + shiftZ1) / scaleZ1 * (1 - deltaZ) + deltaZ

    shiftX2 = torch.tensor(- 1)
    shiftY2 = torch.tensor(1)
    rangeZ2 = 7.90062523 - 0.00062500 # Maximum: 7.90062523, Minimum: 0.00062500
    scaleZ2 = torch.tensor(rangeZ2)
    shiftZ2 = torch.tensor(- 0.00062500)
    x2 = x + shiftX2
    y2 = y + shiftY2
    z2 = x2 ** 2 + y2 ** 2
    z2 = (z2 + shiftZ2) / scaleZ2 * (1 - deltaZ) + deltaZ

    rangeZ = 1.00000000 - 0.07055350 # Maximum: 1.00000000, Minimum: 0.07055350
    scaleZ = torch.tensor(rangeZ)
    shiftZ = torch.tensor(- 0.07055350)

    z = 0.5 * z1 + 0.5 * z2
    z = (z + shiftZ) / scaleZ * (1 - deltaZ) + deltaZ
    return z



def objectiveFunction(x, y):
    if objectiveFunctionName == 'Rosenbrock':
        return rosenbrockFunction(x, y)

    elif objectiveFunctionName == 'SixHumpCamel':
        return sixHumpCamelFunction(x, y)

    elif objectiveFunctionName == 'Sphere':
        return sphereFunction(x, y)

    elif objectiveFunctionName == 'Custom':
        return customFunction(x, y)




print("Checking the existence of the directory to save plotted images.")
if os.path.isdir(plotImagePath):
    print("Okay.")
else:
    print("Making the directory.")
    os.mkdir(plotImagePath)
    print(plotImagePath + " has been created.")




#optimizerDictionary = {'SGD': {}, 'AdaGrad': {}, 'RMSprop': {}, 'Adadelta': {}, 'Adam': {}, 'AdamW': {}, 'RAdam': {}, 'AdaBelief': {}, 'AdaDerivative': {}, 'Lion': {}}
optimizerDictionary = {'SGD': {}, 'AdaGrad': {}, 'RMSprop': {}, 'Adadelta': {}, 'AdamW': {}, 'RAdam': {}, 'AdaBelief': {}, 'AdaDerivative': {}, 'Lion': {}}

optimizerDictionary['SGD']['learningRate'] = 1e-1
optimizerDictionary['AdaGrad']['learningRate'] = 2e-1
optimizerDictionary['RMSprop']['learningRate'] = 2e-2
optimizerDictionary['Adadelta']['learningRate'] = 1e+1
#optimizerDictionary['Adam']['learningRate'] = 1e-1
optimizerDictionary['AdamW']['learningRate'] = 1e-1
optimizerDictionary['RAdam']['learningRate'] = 1e-1
optimizerDictionary['AdaBelief']['learningRate'] = 1e-1
optimizerDictionary['AdaDerivative']['learningRate'] = 1e-1
optimizerDictionary['Lion']['learningRate'] = 5e-2

optimizerDictionary['SGD']['color'] = 'lime'
optimizerDictionary['AdaGrad']['color'] = 'orange'
optimizerDictionary['RMSprop']['color'] = 'purple'
optimizerDictionary['Adadelta']['color'] = 'brown'
#optimizerDictionary['Adam']['color'] = 'olive'
optimizerDictionary['AdamW']['color'] = 'yellow'
optimizerDictionary['RAdam']['color'] = 'green'
optimizerDictionary['AdaBelief']['color'] = 'blue'
optimizerDictionary['AdaDerivative']['color'] = 'red'
optimizerDictionary['Lion']['color'] = 'turquoise'

for key in optimizerDictionary:
    x = torch.tensor(- 0.9, requires_grad=True)
    y = torch.tensor(0.9, requires_grad=True)
    optimizerDictionary[key]['parameters'] = [torch.tensor(- 0.9, requires_grad=True), torch.tensor(0.9, requires_grad=True)]
    optimizerDictionary[key]['xList'] = [optimizerDictionary[key]['parameters'][0].item()]
    optimizerDictionary[key]['yList'] = [optimizerDictionary[key]['parameters'][1].item()]
    optimizerDictionary[key]['zList'] = [objectiveFunction(x, y).item()]

    if key == 'SGD':
        optimizerDictionary['SGD']['optimizer'] = torch.optim.SGD(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['SGD']['learningRate'])
    elif key == 'AdaGrad':
        optimizerDictionary['AdaGrad']['optimizer'] = torch.optim.Adagrad(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['AdaGrad']['learningRate'])
    elif key == 'RMSprop':
        optimizerDictionary['RMSprop']['optimizer'] = torch.optim.RMSprop(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['RMSprop']['learningRate'])
    elif key == 'Adadelta':
        optimizerDictionary['Adadelta']['optimizer'] = torch.optim.Adadelta(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['Adadelta']['learningRate'])
    #elif key == 'Adam':
        #optimizerDictionary['Adam']['optimizer'] = torch.optim.Adam(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['Adam']['learningRate'])
    elif key == 'AdamW':
        optimizerDictionary['AdamW']['optimizer'] = torch.optim.AdamW(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['AdamW']['learningRate'])
    elif key == 'RAdam':
        optimizerDictionary['RAdam']['optimizer'] = torch.optim.RAdam(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['RAdam']['learningRate'],  betas=(0.9, 0.999), eps=1e-16, weight_decay=1e-3)
    elif key == 'AdaBelief':
        optimizerDictionary['AdaBelief']['optimizer'] = AdaBelief(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['AdaBelief']['learningRate'], betas=(0.9, 0.999), eps=1e-16, weight_decay=1e-3, rectify=False)
    elif key == 'AdaDerivative':
        optimizerDictionary['AdaDerivative']['optimizer'] = AdaDerivative.AdaDerivative(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['AdaDerivative']['learningRate'], betas=(0.9, 0.999), eps=1e-16, weight_decay=1e-3, rectify=False)
    elif key == 'Lion':
        optimizerDictionary['Lion']['optimizer'] = Lion(params=optimizerDictionary[key]['parameters'], lr=optimizerDictionary['Lion']['learningRate'], weight_decay=1e-8)




X = numpy.arange(-1, 1, 0.025)
Y = numpy.arange(-1, 1, 0.025)
X, Y = numpy.meshgrid(X, Y)
X = torch.from_numpy(X.astype(numpy.float32)).clone()
Y = torch.from_numpy(Y.astype(numpy.float32)).clone()
Z = objectiveFunction(X, Y)
maximum = torch.max(Z)
print("Maximum: {:.8f}".format(maximum))
minimum = torch.min(Z)
print("Minimum: {:.8f}".format(minimum))
 
figure1 = pyplot.figure(figsize=(12, 10))
figure1.subplots_adjust(left=-0.02, right=0.8, bottom=0.02, top=0.98)
axis1 = figure1.add_subplot(1, 1, 1, projection='3d', elev=45, azim=330, zorder=1)
axis1.xaxis.pane.set_facecolor((0.5, 0.5, 0.5, 0.5))
axis1.yaxis.pane.set_facecolor((0.5, 0.5, 0.5, 0.5))
axis1.zaxis.pane.set_facecolor((0.5, 0.5, 0.5, 0.5))




def update(i):
    print('Step: {:3d}'.format(i + 1))

    axis1.cla()
    plotWireframe1 = axis1.plot_wireframe(X, Y, Z, rstride=2, cstride=2, zorder=1)
    plotSurface1 = axis1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='ocean', norm=LogNorm(), alpha=0.8, zorder=1)

    for key in optimizerDictionary:
        optimizerDictionary[key]['optimizer'].zero_grad()
        outputs = objectiveFunction(optimizerDictionary[key]['parameters'][0], optimizerDictionary[key]['parameters'][1])
        outputs.backward()
        optimizerDictionary[key]['optimizer'].step()
        optimizerDictionary[key]['xList'].append(optimizerDictionary[key]['parameters'][0].item())
        optimizerDictionary[key]['yList'].append(optimizerDictionary[key]['parameters'][1].item())
        optimizerDictionary[key]['zList'].append(outputs.item())
        labelString = key + ': lr ' + str(optimizerDictionary[key]['learningRate'])
        plot1 = axis1.plot(optimizerDictionary[key]['xList'], optimizerDictionary[key]['yList'], optimizerDictionary[key]['zList'], color=optimizerDictionary[key]['color'], marker='o',  linestyle='solid', alpha=0.75, markersize=1, zorder=4)
        plot1 = axis1.plot(optimizerDictionary[key]['xList'][-1], optimizerDictionary[key]['yList'][-1], optimizerDictionary[key]['zList'][-1], color=optimizerDictionary[key]['color'], marker='o',  linestyle='solid', alpha=1.0, markersize=8, zorder=4, label=labelString)

        #axis1.set_xlabel('x', fontsize=8)
        #axis1.set_ylabel('y', fontsize=8)
        #axis1.set_zlabel('z', fontsize=8)

        axis1.set_xlim(- 1, 1)
        axis1.set_ylim(- 1, 1)

        axis1.legend(bbox_to_anchor=(1.25, 1), loc='upper right', fontsize=16)

#pyplot.savefig(plotImagePath + '/{}_optim_{}.png'.format(objectiveFunctionName, key))
animation = FuncAnimation(fig=figure1, func=update, frames=numberOfFrames, interval=100, repeat=False)
animation.save(plotImagePath + '/Optimizers (' + objectiveFunctionName + ').gif')
