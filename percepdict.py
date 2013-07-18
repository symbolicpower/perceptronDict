import os, csv, random, copy
os.chdir("C:\\Python33\\mine\\")

def weights_dictionary(training_file):
    '''Take as input the training file in csv format. Make every enrty
    in every row (except the first row and column) as a key in a dictionary.
    To each assign a value which is the weight for the corresponding input'''

    #initialize the weights dictionary with an entry for the activation node
    weights = {'activation': 2*random.random() - 1}
    try:
        with open(training_file,'r') as csvf:
            csvfr = csv.reader(csvf)
            #skip first row of headers
            next(csvfr)
            for row in csvfr:
                for entry in row[1:]:
                    #initialize with random weights between -1 and 1
                    weights[entry] = 2*random.random() - 1
    except IOError as ioerr:
        print("File error: " + str(ioerr))
    return weights


def train(training_file):
    '''Take as input the training file in csv format. The first column consists
    of the output values for a given set of inputs populated in the rest of the columns.
    Return the updated dictionary of weights.'''

    weights = weights_dictionary(training_file)

    outputs = []
    outputs_previous = [-1]
    iterCount = 1
    for i in range(10):
        print("iteration", iterCount)
        iterCount += 1
        if (outputs == outputs_previous):
            print("Converged")
        outputs_previous = copy.deepcopy(outputs)
        with open(training_file,'r') as csvf:
            csvfr = csv.reader(csvf)
            #skip first row of headers
            next(csvfr)
            for row in csvfr:
                #set activation to the weighted value obtained from the bias input
                activation = -1*weights['activation']
                
                #compute the activation with the current weights
                for entry in row[1:]:
                    activation += weights[entry]

                #Decide whether neuron fires or not
                if (activation > 0):
                    activation = 1
                    outputs.append(1)
                else:
                    activation = 0
                    outputs.append(0)

                #Learning rate
                eta = 0.25

                #Update weights
                weights['activation'] += eta*(int(row[0]) - activation)*(-1)
                for entry in row[1:]:
                    weights[entry] += eta*(int(row[0]) - activation)

    return weights

def predict(training_file, testing_file):
    '''Take as input the training and testing files in csv format. The first column in
    the training file consists of the output values for a given set of inputs populated
    in the rest of the columns. The first column in the testing file consists of serial
    numbers while rest of the columns match with those of the training file.'''

    weights = train(training_file)

    outputs = []
    with open(testing_file,'r') as csvf:
        csvfr = csv.reader(csvf)
        #skip first row of headers
        next(csvfr)
        for row in csvfr:
            #set activation to the weighted value obtained from the bias input
            activation = -1*weights['activation']
            
            #compute the activation with the current weights
            for entry in row[1:]:
                activation += weights[entry]

            #Decide whether neuron fires or not
            if (activation > 0):
                outputs.append(1)
            else:
                outputs.append(0)

    return outputs
