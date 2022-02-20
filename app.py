from fetch_prices import load
from processing import computeWithFiboHistory, fibanacciSequence, movingAverage, pairHistoryMovingAverage, pairHistoryMovingAverageRange
from training_data_loader import loadTrainingData
from visualize import plotMultiPairHistory, plotPairHistory
import numpy as np

trainData = loadTrainingData()
print(trainData)
