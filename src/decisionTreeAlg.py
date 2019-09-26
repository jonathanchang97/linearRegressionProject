import numpy as np
import math


class Node:
  def __init__(self, feature, value, branches):
    self.feature = feature
    self.value = value
    self.branches = branches


class Leaf:
  def __init__(self, guess, value, feature):
    self.guess = guess
    self.value = value
    self.feature = feature


def getInputs():
	while True:
		try:
			trainingSize = int(input("Please enter a training set size " +
				"(a positive multiple of 250 that is <= 1000): \n"))
		except ValueError:
			print("Sorry, that was an invalid input. Please try again")
			exit(1)
		else:
			if trainingSize == 250 or trainingSize == 500 \
			   or trainingSize == 1000:
				break
			else:
				print("Sorry, The training set size must be 250, 500, or 1000.")
				print("Please try again")
				exit(1)

	while True:
		try:
			trainingIncr = int(input("Please enter a training increment " +
				"(either 10, 25, or 50): \n"))
		except ValueError:
			print("Sorry, that was an invalid input. Please try again")
			exit(1)
		else:
			if trainingIncr == 10 or trainingIncr == 25 \
			   or trainingIncr == 50:
				break
			else:
				print("Sorry, The training set increment must be 10, 25, or 50")
				print("Please try again")
				exit(1)

	while True:
		heur = input("Please enter a heuristic to use (either " +
			"[C]ounting-based or [I]nformation theoretic): \n")
		if heur == "C" or heur == "I":
			break
		else:
			print("Sorry, you must enter either 'C' or 'I'")
			print("Please try again")
			exit(1)

	return trainingSize, trainingIncr, heur


def loadData():
	print("\nLoading Property Information from file.")
	properties = np.loadtxt(fname="../input_files/properties.txt", 
							 dtype=str, delimiter=': ')

	print("Loading Data from database.\n")
	dataSet = np.loadtxt(fname="../input_files/mushroom_data.txt", dtype=str)

	propertyIndices = np.zeros((22,1), dtype=int)
	for x in range(0,21):
		propertyIndices[x,0] = x

	properties = np.append(properties, propertyIndices, axis=1)
	return properties, dataSet


def selectSubset(dataSet, subsetSize):
	subsetIndices = np.random.choice(dataSet.shape[0], 
									 subsetSize, replace=False)

	subset = dataSet[subsetIndices,:]
	dataSet = np.delete(dataSet, subsetIndices, axis=0)
	return subset, dataSet


def guessMostFrequent(data):
	edibleCount = 0;
	poisonCount = 0;
	for mush in data:
		if mush[22] == "e":
			edibleCount += 1
		else:
			poisonCount += 1

	if edibleCount == data.shape[0]:
		return "e", True
	elif poisonCount == data.shape[0]:
		return "p", True
	elif edibleCount > poisonCount:
		return "e", False
	else:
		return "p", False


def mostImportantC(remFeatures, data):
	topScore = (0, None)
	for feature in remFeatures:
		# find the subsets of data given the feature
		values = feature[1].split()

		eSubsetTotal = np.where(data[:,22] == "e")
		pSubsetTotal = np.where(data[:,22] == "p")
		totalScore = 0

		for v in values:
			eSubset = np.where(data[eSubsetTotal,int(feature[2])] == v)
			pSubset = np.where(data[pSubsetTotal,int(feature[2])] == v)

			if eSubset[0].shape[0] > pSubset[0].shape[0]:
				totalScore += eSubset[0].shape[0]
			else:
				totalScore += pSubset[0].shape[0]

		# keep the feature with the highest total score, which is the
		# greatest correlation to edible and poison mushrooms
		if totalScore > topScore[0]:
			topScore = (totalScore, feature)

	return topScore[1]


def mostImportantI(remFeatures, data):
	topScore = (0, None)
	for feature in remFeatures:
		# find the subsets of data given the feature
		values = feature[1].split()

		eSubsetTotal = np.where(data[:,22] == "e")
		pSubsetTotal = np.where(data[:,22] == "p")
		pt = eSubsetTotal[0].shape[0]
		nt = pSubsetTotal[0].shape[0]
		tt = nt + pt

		G = 0
		totalR = 0
		for v in values:
			eSubset = np.where(data[eSubsetTotal,int(feature[2])] == v)
			pSubset = np.where(data[pSubsetTotal,int(feature[2])] == v)

			p = eSubset[0].shape[0]
			n = pSubset[0].shape[0]
			t = p + n

			# H = -P(Pos)log2P(Pos) + P(Neg)log2P(Neg)
			if t == 0:
				break
			if p == 0:
				H = -(n / t) * math.log2(n / t)
			elif n == 0:
				H = -(p / t) * math.log2(p / t)
			else:
				H = -((p / t) * math.log2(p / t) + (n / t) * math.log2(n / t))	
			totalR += (t / tt) * H

		H = -((pt / tt) * math.log2(pt / tt) + (nt / tt) * math.log2(nt / tt))
		G = H - totalR

		# keep the feature with the highest total score, which is the
		# greatest correlation to edible and poison mushrooms
		if G > topScore[0]:
			topScore = (G, feature)

	return topScore[1]


def decisionTreeTrain(data, value, currFeature, remFeatures, parentGuess, heur):
	if data.shape[0] == 0:
		return Leaf(parentGuess, value, currFeature)
	
	guess, unambiguous = guessMostFrequent(data)

	if unambiguous == True or remFeatures.shape[0] == 0:
		return Leaf(guess, value, currFeature)
	else:
		# basic counting
		if heur == "C":
			feature = mostImportantC(remFeatures, data)
		# information theoretic
		else:
			feature = mostImportantI(remFeatures, data)
		values = feature[1].split()
		
		branches = []
		for v in values:
			newData = np.where(data[:,int(feature[2])] == v)
			deleteIndex = np.where(remFeatures[:,0] == feature[0])
			newNode = decisionTreeTrain(data[newData[0],:], v, feature,
										np.delete(remFeatures, deleteIndex, 0),
										guess, heur)
			branches.append(newNode)

		return Node(currFeature, value, branches)


def decisionTreeTest(tree, testPoint):
	if isinstance(tree, Leaf):
		return tree.guess
	else:
		for branch in tree.branches:
			if testPoint[int(branch.feature[2])] == branch.value:
				return decisionTreeTest(branch, testPoint)


def printTree(tree, count, printStr, fst):
	if fst == True:
		for branch in tree.branches:
			count = printTree(branch, count, printStr, False)
			count += 1
	elif isinstance(tree, Node):
		printStr += "Attrib #{}: {}; ".format(tree.feature[2], tree.value)
		for branch in tree.branches:
			count = printTree(branch, count, printStr, False)
			count += 1
		return count
	else:
		branchStr = "Branch[{}]: ".format(count)
		printStr += "Attrib #{}: {}; {}.".format(tree.feature[2], 
												 tree.value, tree.guess)
		print(branchStr + printStr)
		return count
		
	
def main():
	print("Welcome to the decision tree learning algorithm.")	
	print("Please enter the following inputs\n")

	# get inputs from user
	trainingSize, trainingIncr, heur = getInputs();

	# load in data
	properties, fullDataset = loadData()
	
	# Run decision tree algorithm for each training set increment till the
	# entire training set is used

	# select training data set and delete from full data set
	print("Collecting set of " + str(trainingSize) + " training examples.\n")
	trainingSet, trimmedDataSet = selectSubset(fullDataset, trainingSize)

	incrList = range(trainingIncr, trainingSize+1, trainingIncr)
	successRateList = []
	for currTrainingIncr in incrList:
		# select training examples from data set based on increment
		print("Running with " + str(currTrainingIncr) + 
			  " examples in training set.\n")
		(exampleSet,_) = selectSubset(trainingSet, currTrainingIncr)

		# build decision tree using the example set from the training set
		tree = decisionTreeTrain(exampleSet, None, None, properties, 
			                     "Poison", heur)

		count = 0
		for testPoint in trimmedDataSet:
			guess = decisionTreeTest(tree, testPoint)
			if guess == testPoint[22]:
				count +=1

		successRate = count / trimmedDataSet.shape[0] * 100
		successRateList.append(successRate)
		print("Given current tree, there are " + str(count) + " correct " +
			  "classifications out of " + str(trimmedDataSet.shape[0]) + 
			  " possible (a success rate of {0:.4f} percent).\n".
			  format(successRate))

	# Print Out final results
	print("-------------------")
	print("Final Decision Tree")
	print("-------------------\n")

	printTree(tree, 0, "", True)

	print("-------------------")
	print("Statistics")
	print("-------------------\n")

	count = 0
	for incr in incrList:
		print("Training set size: {0}.  Success:  {1:.4f} percent."
			  .format(incr, successRateList[count]))
		count += 1


if __name__ == '__main__':
    main()
