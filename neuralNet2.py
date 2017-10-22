import numpy as np
import math
import random
import dill
import pandas as pd


#DEFINE CLASSES NEURON, LAYER, NETWORK
class Neuron:
    def __init__(self,numberInputs):
        self.numberInputs=numberInputs
        self.inWeights=np.array([])
        self.randomRange=.31
        for x in range(0,numberInputs):
            self.inWeights=np.append(self.inWeights,np.random.uniform(-self.randomRange,self.randomRange))

    # Since the output data is a percent (between 0-1) the sigmoid will likely be used
    def tanh(self, x, deriv=False):
        nonlin = np.tanh(x)
        if (deriv == True):
            return (1 - np.square(nonlin))
        return nonlin

    def sigmoid(self, x, deriv=False):
        nonlin = 1 / (1 + np.exp(-x))
        if (deriv == True):
            return nonlin * (1 - nonlin)
        return nonlin

    def evaluate(self,inputValues):
        #print self.inWeights.shape , inputValues
        self.summation=np.dot(self.inWeights,inputValues)
        self.value= self.sigmoid(self.summation) #+ self.weights[-1] Bias Nodes? Weights and Values?

    def setInWeights(self,inWeights):
        self.inWeights=inWeights

    def setOutWeights(self,outWeights):
        self.outWeights=outWeights

    def setError(self,errorsFor,weightsFor):
        self.error=np.dot(errorsFor, weightsFor)

class Layer:
    def __init__(self, name, numberNeuronsInLayer,numberInputsPerNeuron):  #layer1 inputs are parameters of study, but layer2 inputs are outputs of layer 1
        self.name=name
        self.numberNeuronsInLayer=numberNeuronsInLayer
        self.neurons=np.array([])
        for x in range(0,numberNeuronsInLayer):
            self.neurons=np.append(self.neurons,Neuron(numberInputsPerNeuron))

class Network:
    def __init__(self,numberParameters, numberHiddenLayers, numberNeurons, numberOutputs):
        self.numberHiddenLayers=numberHiddenLayers
        self.numberParameters=numberParameters
        self.numberNeuronsPer=np.array([])
        self.network=np.array([])
        self.numberOutputs=numberOutputs

        for x in range(0,numberHiddenLayers):
            self.numberNeuronsPer=np.append(self.numberNeuronsPer, numberNeurons[x])
            if x == 0:
                self.network=np.append(self.network,Layer("layer%s" %(x + 1), self.numberNeuronsPer[x].astype(np.int64), numberParameters))
            else:
                self.network = np.append(self.network, Layer("layer%s" %(x + 1), self.numberNeuronsPer[x].astype(np.int64), self.numberNeuronsPer[x-1].astype(np.int64)))

        self.network=np.append(self.network, Layer("output",numberOutputs,self.numberNeuronsPer[-1].astype(np.int64)))

        self.numberNeuronsPer=np.append(self.numberNeuronsPer,numberOutputs)

#DEFINE INITIALIZATION OF NETWORK
def initializeNetwork(numberParameters, numberHiddenLayers, numberNeurons, numberOutputs, priorInWeights=False):
    return Network(numberParameters,numberHiddenLayers,numberNeurons, numberOutputs)

#DEFINE FEEDFOWARD FUNCTION TO EVALUATE NEURONS
def feedfoward(theNetwork, parameterValues):
    valueArray = [np.array(parameterValues)]

    for l in range (0,len(theNetwork.network)):
        valueArray.append(np.array([]))
        for n in range (0,len(theNetwork.network[l].neurons)):
            theNetwork.network[l].neurons[n].evaluate(valueArray[l][:])
            valueArray[l + 1] =np.append(valueArray[l+1],theNetwork.network[l].neurons[n].value)

    #print valueArray
    return valueArray[-1]

#ASSIGN OUTWEIGHTS TO HELP WITH BACKPROPAGAION OF ERROR
def assignOutWeights(theNetwork):
    for l in range(0,len(theNetwork.network)-1):
        for n in range(0,len(theNetwork.network[l].neurons)):
            outWeights_n=np.array([])
            for m in range(0,len(theNetwork.network[l+1].neurons)):
                outWeights_n=np.append(outWeights_n,theNetwork.network[l+1].neurons[m].inWeights[n])

            #print outWeights_n
            theNetwork.network[l].neurons[n].setOutWeights(outWeights_n)

#DEFINE FUNCTION TO ASSIGN ERROR TO EACH NEURON FOR USE IN CHANGING WEIGHTS (LEARNING)
def backProp(theNetwork,correctAnswer):
    deltaEnd=correctAnswer-theNetwork.network[-1].neurons[0].value
    theNetwork.network[-1].neurons[0].setError(deltaEnd,1)
    errorArray=[np.array(deltaEnd)]

    for l in range(-2,-len(theNetwork.network)-1,-1):
        errorArray.insert(0,np.array([]))
        for n in range(0, len(theNetwork.network[l].neurons)):
            theNetwork.network[l].neurons[n].setError(errorArray[l+1],theNetwork.network[l].neurons[n].outWeights)
            errorArray[l]=np.append(errorArray[l],theNetwork.network[l].neurons[n].error)

#CHANGE THE WEIGHTS SO THE NETWORK LEARNS!!!
def changeWeights(theNetwork,learningCurve,parameterValues):

    #Check weights before
    #for l in range(0, len(theNetwork.network)):
    #    for n in range(0,len(theNetwork.network[l].neurons)):
    #        print theNetwork.network[l].neurons[n].inWeights


    for l in range(0, len(theNetwork.network)):
        for n in range(0,len(theNetwork.network[l].neurons)):
            newInWeights=np.array([])
            error=theNetwork.network[l].neurons[n].error
            derivative = theNetwork.network[l].neurons[n].sigmoid(theNetwork.network[l].neurons[n].summation, True)

            for i in range(0,theNetwork.network[l].neurons[n].numberInputs):
                oldWeight=theNetwork.network[l].neurons[n].inWeights[i]

                if l==0:
                    x_input=parameterValues[i]
                else:
                    x_input=theNetwork.network[l-1].neurons[i].value

                newInWeight_n=oldWeight+learningCurve*error*derivative*x_input
                newInWeights=np.append(newInWeights,newInWeight_n)

            theNetwork.network[l].neurons[n].setInWeights(newInWeights)

    #Check weights after
    #for l in range(0, len(theNetwork.network)):
    #    for n in range(0,len(theNetwork.network[l].neurons)):
    #        print theNetwork.network[l].neurons[n].inWeights


#FEED TRAINING DATA
def trainNetwork(theNetwork, data, learningCurve):
    #Assumes goal is last column of data, assumes data is normalized
    numDataPoints=len(data)
    parameters=data[:,:-1]
    goals=data[:,-1]
    networkGuess=np.array([])

    for p in range(0,numDataPoints):
        feedfoward(theNetwork, parameters[p, :])
        networkGuess=np.append(networkGuess,feedfoward(theNetwork,parameters[p,:]))
        assignOutWeights(theNetwork)
        backProp(theNetwork,goals[p])     #goal data[p,-1]
        changeWeights(theNetwork,learningCurve,parameters[p,:])
        feedfoward(theNetwork, parameters[p, :])

#CHECK HOW GOOD IT IS
def checkNetworkGuesses(theNetwork, data):
    networkGuess=[]
    for p in range(0,len(data)):
        networkGuess.append(np.asscalar(feedfoward(theNetwork,data[p,:-1])))

    return np.mean(np.absolute(data[:,-1]-networkGuess)) #,np.mean(np.absolute(data[:,1]-networkGuess))

def saveNetwork(theNetwork):
    answer=raw_input("Name the network (layers, neurons per, what training data, learning curve): ")
    with open("%s" %(answer),'wb') as f:
        dill.dump(theNetwork,f)

def importNetwork(answer):
    #answer=raw_input("What's the name of the network you would like to import? ")
    with open("%s" %(answer),'rb') as f:
        theNetwork=dill.load(f)
    return theNetwork

def importData(file):
    df=pd.read_excel(file,sheetname="final", header=None)
    data=df.as_matrix()
    numDataPoints=len(data)
    numParametersPlusGoal = len(data[0])
    dataMeans = np.array([])
    dataSig = np.array([])

    for k in range(0, len(data[0])):
        dataMeans = np.append(dataMeans, np.mean(data[:,k]))
        dataSig = np.append(dataSig, np.std(data[:,k]))

        #Rewrite Mean and Sig of binary data
        if data[0,k]==-1 or data[0,k]==1 or data[0,k]==0:
            dataMeans[k]=0
            dataSig[k]=1

    dataNormed=np.empty(shape=(numDataPoints,numParametersPlusGoal))

    for p in range(0,len(data)):
        dataNormed[p]=(data[p]-dataMeans)/dataSig

    #Overwrite data the doesn't need normalization
    dataNormed[:,27]=data[:,27]

    return dataNormed,dataMeans,dataSig

###       OPTIMIZATION CODE         #####
dataNormed=importData("testDATA.xlsx")
numParameters=len(dataNormed[0])-1
numberOfOutputs =1

results=[]
def optimalNetwork():
    for a in range(2,5):
        numberHiddenLayers=a
        numberNeurons = []
        for b in range(20,85,5):
            neuronsValue=b
            for c in range(0, numberHiddenLayers):
                numberNeurons.append(neuronsValue)
            for d in np.arange(.5,5,.5):
                learningCurve=d

                firstNetwork = initializeNetwork(numParameters, numberHiddenLayers, numberNeurons, numberOfOutputs)
                trainNetwork(firstNetwork, dataNormed, learningCurve)
                ans=checkNetworkGuesses(firstNetwork, dataNormed)

                results.append([a,b,d,ans])

    scores=[item[-1] for item in results]
    bestCombo=results[scores.index(min(scores))]
    return bestCombo

#BEST COMBO 2 LAYERS, 25 NEURONS PER, 4.5 LEARNING CURVE
#print optimalNetwork()


####        FIRE UP TRAINING! SET & SAVE WHAT YOU WANT      ####
"""
dataNormed=importData("testDATA.xlsx")
numParameters=len(dataNormed[0])-1  #drop the goals
numberHiddenLayers=2
neuronsValue=25
numberNeurons=[]
for l in range(0, numberHiddenLayers):
    numberNeurons.append(neuronsValue)
numberOfOutputs =1
learningCurve=4.5

firstNetwork = initializeNetwork(numParameters,numberHiddenLayers,numberNeurons, numberOfOutputs)
trainNetwork(firstNetwork,dataNormed,learningCurve)
print checkNetworkGuesses(firstNetwork,dataNormed)
saveNetwork(firstNetwork)
"""

####        GET NETWORK GUESS FROM SAVED NETWORK & NEW DATA         #####
dataTraining, trainingMean, trainingSig=importData("testDATA.xlsx")
firstNetwork=importNetwork("network2_25_4p5_19dataPts")
print feedfoward(firstNetwork,dataNormed[0,:-1])










######     OLD CODE USED TO CHECK STUFF      ########

#CHECK IN OUT WEIGHTS MATCH (TRANSPOSED)
"""
for l in range(1,3):
    for n in range(0,len(firstNetwork.network[l].neurons)):
        print firstNetwork.network[l].neurons[n].inWeights
"""



# TEST SIMPLE NETWORK. WORKS WELL WITH: 10NEURONS PER, TANH, RANDOM WEIGHTS -.3 TO .3, LEARNING CURVE=1
"""  

x = np.array([[1, 0, 0, 0.20310556,	0.04797283,	0.02653476,	1.60488704, -.8, 0, 1, .3]])
firstNetwork = initializeNetwork(10, 2, 1)
for i in range(0,6):
    feedfoward(firstNetwork,[1, 0, 0, 0.20310556,	0.04797283,	0.02653476,	1.60488704, -.8, 0, 1])
    assignOutWeights(firstNetwork)
    backProp(firstNetwork,.3)     #goal data[p,-1]
    changeWeights(firstNetwork,.8,[1, 0, 0, 0.20310556,	0.04797283,	0.02653476,	1.60488704, -.8, 0, 1])
"""


"""
for l in range(0,len(firstNetwork.network)):
    for n in range(0,len(firstNetwork.network[l].neurons)):
        print firstNetwork.network[l].neurons[n].inWeights
        print firstNetwork.network[l].neurons[n].value

trainNetwork(firstNetwork,x,.2)
for l in range(0,len(firstNetwork.network)):
    for n in range(0,len(firstNetwork.network[l].neurons[n])):
        print firstNetwork.network[l].neurons[n].inWeights
        print firstNetwork.network[l].neurons[n].value
"""





