import numpy as np
import math
import random
import dill


#DEFINE CLASSES NEURON, LAYER, NETWORK
class Neuron:
    def __init__(self,numberInputs):
        self.numberInputs=numberInputs
        self.inWeights=np.array([])
        for x in range(0,numberInputs):
            self.inWeights=np.append(self.inWeights,np.random.uniform(0,1))

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
        self.summation=np.dot(self.inWeights[:], np.transpose(inputValues))
        self.value= self.sigmoid(self.summation) #+ self.weights[-1] Bias Nodes? Weights and Values?

    def setInWeights(self,inWeights):
        self.inWeights=inWeights

    def setOutWeights(self,outWeights):
        self.outWeights=outWeights

    def setError(self,errorsFor,weightsFor):
        self.error=np.dot(errorsFor, weightsFor)


#    def __str__(self):
#            return 'Weights: %s' %( str(self.weights[:-1]),str(self.weights[-1]) )


class Layer:
    def __init__(self, name, numberNeuronsInLayer,numberInputsPerNeuron):  #layer1 inputs are parameters of study, but layer2 inputs are outputs of layer 1
        self.name=name
        self.numberNeuronsInLayer=numberNeuronsInLayer
        self.neurons=np.array([])
        for x in range(0,numberNeuronsInLayer):
            self.neurons=np.append(self.neurons,Neuron(numberInputsPerNeuron))

#    def __str__(self):
#        return 'Layer:\n\t' + '\n\t'.join([str(neuron) for neuron in self.neurons]) + ''

class Network:
    def __init__(self,numberParameters, numberHiddenLayers, numberOutputs):
        self.numberHiddenLayers=numberHiddenLayers
        self.numberParameters=numberParameters
        #self.parameterValues=parameterValues
        self.numberNeuronsPer=np.array([])
        self.network=np.array([])
        self.numberOutputs=numberOutputs

        for x in range(0,numberHiddenLayers):
            self.numberNeuronsPer=np.append(self.numberNeuronsPer, input("Input the number of neurons for layer %s: " %(x+1)))
            if x == 0:
                self.network=np.append(self.network,Layer("layer%s" %(x + 1), self.numberNeuronsPer[x].astype(np.int64), numberParameters))
            else:
                self.network = np.append(self.network, Layer("layer%s" %(x + 1), self.numberNeuronsPer[x].astype(np.int64), self.numberNeuronsPer[x-1].astype(np.int64)))

        self.network=np.append(self.network, Layer("output",numberOutputs,self.numberNeuronsPer[-1].astype(np.int64)))

        self.numberNeuronsPer=np.append(self.numberNeuronsPer,numberOutputs)


#DEFINE NORMALIZATION TOOLS

def gaussianNorm(vector):
    return (vector-np.mean(vector))/np.std(vector)

def featureScaling(vector):
    return (vector-np.amin(vector))/(np.amax(vector)-np.amin(vector))

#DEFINE INITIALIZATION OF NETWORK
def initializeNetwork(numberParameters, numberHiddenLayers, numberOutputs, priorInWeights=False):
    theNetwork=Network(numberParameters,numberHiddenLayers,numberOutputs)
    """ I think it's better to save the whole network object, thus this isn't needed
    if priorInWeights!=False:
        if len(theNetwork.network)!=len(priorInWeights):
            print "Number of Hidden Layers inputed doesn't match the weights array given"
        else:
            for l in range(0,numberHiddenLayers+1):
                if len(theNetwork.network[l].neurons[0])!=len(priorInWeights[l][0]):
                    print "Number of neurons in the %s layer doesn't match the weights array given" %(l)

        #IF THE WEIGHTS DATA MATCHES THE NETWORK SETUP....THEN ASSIGN THE WEIGHTS
        for l in range(0,numberHiddenLayers+1):
            for n in range(theNetwork.network[l].neurons[n]):
                theNetwork.network[l].neurons[n].setInWeights(priorInWeights[l][n])
    """

    return theNetwork


#DEFINE FEEDFOWARD FUNCTION TO EVALUATE NEURONS
def feedfoward(theNetwork, parameterValues):
    valueArray=np.array([parameterValues,[],[],[]])

    for l in range (0,len(theNetwork.network)):
        for n in range (0,len(theNetwork.network[l].neurons)):
            theNetwork.network[l].neurons[n].evaluate(valueArray[l][:])
            valueArray[l+1]=np.append(valueArray[l+1],theNetwork.network[l].neurons[n].value)

    print valueArray

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

    errorArray=np.array([[],[],deltaEnd])

    for l in range(-2,-len(theNetwork.network)-1,-1):
        for n in range(0, len(theNetwork.network[l].neurons)):
            theNetwork.network[l].neurons[n].setError(errorArray[l+1],theNetwork.network[l].neurons[n].outWeights)
            errorArray[l]=np.append(errorArray[l],theNetwork.network[l].neurons[n].error)

    #print errorArray

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
def trainNetwork(theNetwork, parameterValues, goal, learningCurve):
    feedfoward(theNetwork,parameterValues)
    assignOutWeights(theNetwork)
    backProp(theNetwork,goal)
    changeWeights(theNetwork,learningCurve,parameterValues)

    answer=raw_input("Do you want to save the Network? y/n: ")
    if answer=="n":
        answer2=input("Are you sure? y/n: ")
        if answer2=="y":
            return 0
        else:
            answer3=raw_input("Name the network file (layers, neurons per layer, thru what training data, learning curve)")
            with open("%s" %(answer3),'wb') as f:
                dill.dump(theNetwork,f)
    else:
        answer3 = raw_input("Name the network file (layers, neurons per layer, thru what training data, learning curve)")
        with open("%s" % (answer3), 'wb') as f:
            dill.dump(theNetwork, f)


def importNetwork():
    answer=raw_input("What's the name of the network you would like to import? ")
    with open("%s" %(answer),'rb') as f:
        theNetwork=dill.load(f)

    return theNetwork

# INITIALIZE THE NETWORK THAT WILL BE USED
"""
x = np.array([1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
firstNetwork = initializeNetwork(10, 2, 1)
trainNetwork(firstNetwork,x,.9,20)
"""

firstNetwork=importNetwork()
print firstNetwork.network



#CHECK IN OUT WEIGHTS MATCH (TRANSPOSED)
"""
for l in range(1,3):
    for n in range(0,len(firstNetwork.network[l].neurons)):
        print firstNetwork.network[l].neurons[n].inWeights
"""






