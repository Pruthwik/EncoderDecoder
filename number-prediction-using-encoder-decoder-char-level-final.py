'''
Sequence to sequence example in Keras (character-level).
This is an implementation of a basic character-level 
sequence-to-sequence model. 
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from random import sample
from nltk.translate import bleu_score


def readLinesFromFile(filePath):
    with open(filePath, 'r', encoding='utf-8') as f:
        return f.read().split('\n')


def splitDataIntoInputAndTarget(lines, numSamples):
    inputTexts, targetTexts = list(), list()
    inputCharacters, targetCharacters = set(), set()
    for line in lines[: min(numSamples, len(lines) - 1)]:
        inputText, targetText = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        targetText = '\t' + targetText + '\n'
        inputTexts.append(inputText)
        targetTexts.append(targetText)
        for char in inputText:
            if char not in inputCharacters:
                inputCharacters.add(char)
        for char in targetText:
            if char not in targetCharacters:
                targetCharacters.add(char)
    return inputTexts, targetTexts, inputCharacters, targetCharacters


def vectorizeInputData(inputTexts, inputCharacters):
    inputCharacters = sorted(list(inputCharacters))
    numOfInTokens = len(inputCharacters)
    maxSeqLength = max([len(txt) for txt in inputTexts])
    inputTokenIndex = dict([(char, i) for i, char in enumerate(inputCharacters)])
    inputData = np.zeros((len(inputTexts), maxSeqLength, numOfInTokens), dtype='float32')
    for index, inputText in enumerate(inputTexts):
        for i, char in enumerate(inputText):
            inputData[index, i, inputTokenIndex[char]] = 1.
        inputData[index, i + 1:, inputTokenIndex[' ']] = 1.
    return inputData, inputTokenIndex, maxSeqLength


def vectorizeTestData(testTexts, maxSeqLength, inputTokenIndex):
    testData = np.zeros((len(testTexts), maxSeqLength, len(inputTokenIndex)), dtype='float32')
    for index, testText in enumerate(testTexts):
        for i, char in enumerate(testText):
            testData[index, i, inputTokenIndex[char]] = 1.
        testData[index, i + 1:, inputTokenIndex[' ']] = 1.
    return testData


def vectorizeTargetData(outputTexts, outputCharacters, typeData=0):
    '''
    typeData=0 for decoder inputs and 1 for target output
    decoder inputs are necessary for teacher forcing
    '''
    outputCharacters = sorted(list(outputCharacters))
    numOfOutTokens = len(outputCharacters)
    maxSeqLength = max([len(txt) for txt in outputTexts])
    outputTokenIndex = dict([(char, i) for i, char in enumerate(outputCharacters)])
    outputData = np.zeros((len(outputTexts), maxSeqLength, numOfOutTokens), dtype='float32')
    for index, outputText in enumerate(outputTexts):
        for i, char in enumerate(outputText):
            if not typeData:
                outputData[index, i, outputTokenIndex[char]] = 1.
            elif typeData and i > 0:
                outputData[index, i - 1, outputTokenIndex[char]] = 1.
        if not typeData:
            outputData[index, i + 1:, outputTokenIndex[' ']] = 1.
        else:
            outputData[index, i:, outputTokenIndex[' ']] = 1.
    return outputData, outputTokenIndex, maxSeqLength


def createReverseDictionary(dictItems):
    return {value: key for key, value in dictItems.items()}


def trainModel(inputData, decoderInputData, targetOutputData, maxSeqLengthInput, maxSeqLengthOutput, numberOfInputChars, numberOfOutputChars):
    batchSize, latentDim, epochs = 32, 100, 100
    encoderInputs = Input(shape=(None, numberOfInputChars))
    encoder = LSTM(latentDim, return_state=True)
    encoderOutputs, stateH, stateC = encoder(encoderInputs)
    # We discard 'encoder_outputs' and only keep the states.
    encoderStates = [stateH, stateC]
    # Set up the decoder, using 'encoder_states' as initial state.
    decoderInputs = Input(shape=(None, numberOfOutputChars))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoderLSTM = LSTM(latentDim, return_sequences=True, return_state=True)
    decoderOutputs, _, _ = decoderLSTM(decoderInputs, initial_state=encoderStates)
    decoderDense = Dense(numberOfOutputChars, activation='softmax')
    decoderOutputs = decoderDense(decoderOutputs)

    # Define the model that will turn
    # 'encoder_input_data' & 'decoder_input_data' into 'decoder_target_data'
    model = Model([encoderInputs, decoderInputs], decoderOutputs)

    # Run training
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit([inputData, decoderInputData], targetOutputData,
              batch_size=batchSize,
              epochs=epochs,
              validation_split=0.2)
    # Save model
    model.save('model-enc-dec-num-pred-adam.h5')

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoderModel = Model(encoderInputs, encoderStates)

    decoderStateInputH = Input(shape=(latentDim,))
    decoderStateInputC = Input(shape=(latentDim,))
    decoderStatesInputs = [decoderStateInputH, decoderStateInputC]
    decoderOutputs, stateH, stateC = decoderLSTM(
        decoderInputs, initial_state=decoderStatesInputs)
    decoderStates = [stateH, stateC]
    decoderOutputs = decoderDense(decoderOutputs)
    decoderModel = Model(
        [decoderInputs] + decoderStatesInputs,
        [decoderOutputs] + decoderStates)
    return model, encoderModel, decoderModel


def decodeSequence(inputSeq, encoderModel, decoderModel, numberOfOutputChars, targetTokenIndex, reverseTargetCharIndex, maxSeqLength):
    # Encode the input as state vectors.
    statesValue = encoderModel.predict(inputSeq)

    # Generate empty target sequence of length 1.
    targetSeq = np.zeros((1, 1, numberOfOutputChars))
    # Populate the first character of target sequence with the start character.
    targetSeq[0, 0, targetTokenIndex['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stopCondition = False
    decodedSentence = ''
    while not stopCondition:
        outputTokens, h, c = decoderModel.predict(
            [targetSeq] + statesValue)

        # Sample a token
        sampledTokenIndex = np.argmax(outputTokens[0, -1, :])
        sampledChar = reverseTargetCharIndex[sampledTokenIndex]
        decodedSentence += sampledChar

        # Exit condition: either hit max length
        # or find stop character.
        if (sampledChar == '\n' or len(decodedSentence) > maxSeqLength):
            stopCondition = True
        # Update the target sequence (of length 1).
        targetSeq = np.zeros((1, 1, numberOfOutputChars))
        targetSeq[0, 0, sampledTokenIndex] = 1.
        # Update states
        statesValue = [h, c]
    return decodedSentence


def testOnInDomainDataAndEvaluate(numberOfSamples, sampleSize, inputData, targetTexts, encoderModel, decoderModel, targetCharacters, targetTokenIndex, targetIndexToken, maxTgtSeqLength):
    print('Testing on In-Domain Data')
    inDomainSampleIndices = sample(range(numberOfSamples), sampleSize)
    outputWithPredictions = list()
    averageInDomainBLEUScore = np.zeros(sampleSize,)
    for index, seqIndex in enumerate(inDomainSampleIndices):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        inputSeq = inputData[seqIndex: seqIndex + 1]
        decodedSentence = decodeSequence(inputSeq, encoderModel, decoderModel, len(targetCharacters), targetTokenIndex, targetIndexToken, maxTgtSeqLength)
        goldOutput = targetTexts[seqIndex].strip()
        bleuScore = bleu_score.sentence_bleu([goldOutput.split()], decodedSentence.split())
        averageInDomainBLEUScore[index] = bleuScore
        outputWithPredictions.append('Sentence ' + str(index + 1))
        outputWithPredictions.append('Gold Output : ' + goldOutput)
        outputWithPredictions.append('Predicted Output : ' + decodedSentence.strip())
        outputWithPredictions.append('BLEU SCORE : ' + str(bleuScore))
        outputWithPredictions.append('')
    print('AVG BLEU SCORE :', np.mean(averageInDomainBLEUScore))
    outputWithPredictions.append('\nAverage BLEU Score : ' + str(np.mean(averageInDomainBLEUScore)) + '\n')
    writeListToFile('in-domain-preds-epochs100.txt', outputWithPredictions)


def testOnOutDomainDataAndEvaluate(inputData, targetTexts, encoderModel, decoderModel, targetCharacters, targetTokenIndex, targetIndexToken, maxTgtSeqLength):
    print('Testing on Out-Domain Data')
    outputWithPredictions = list()
    numberOfSamples = inputData.shape[0]
    averageInDomainBLEUScore = np.zeros(numberOfSamples,)
    for index in range(numberOfSamples):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        inputSeq = inputData[index].reshape((-1, inputData[index].shape[0], inputData[index].shape[1]))
        decodedSentence = decodeSequence(inputSeq, encoderModel, decoderModel, len(targetCharacters), targetTokenIndex, targetIndexToken, maxTgtSeqLength)
        goldOutput = targetTexts[index].strip()
        bleuScore = bleu_score.sentence_bleu([goldOutput.split()], decodedSentence.split())
        averageInDomainBLEUScore[index] = bleuScore
        outputWithPredictions.append('Sentence ' + str(index + 1))
        outputWithPredictions.append('Gold Output : ' + goldOutput)
        outputWithPredictions.append('Predicted Output : ' + decodedSentence.strip())
        outputWithPredictions.append('BLEU SCORE : ' + str(bleuScore))
        outputWithPredictions.append('')
    print('AVG BLEU SCORE :', np.mean(averageInDomainBLEUScore))
    outputWithPredictions.append('\nAverage BLEU Score : ' + str(np.mean(averageInDomainBLEUScore)) + '\n')
    writeListToFile('out-domain-preds-epochs100.txt', outputWithPredictions)


def main():
    np.random.seed(100)
    dataPath = 'allNumbers.txt'
    inputLines = readLinesFromFile(dataPath)
    numberOfSamples = 10000
    inputTexts, targetTexts, inputCharacters, targetCharacters = splitDataIntoInputAndTarget(inputLines, numberOfSamples)
    inputData, inputTokenIndex, maxInSeqLength = vectorizeInputData(inputTexts, inputCharacters)
    decoderInputs, targetTokenIndex, maxTgtSeqLength = vectorizeTargetData(targetTexts, targetCharacters, 0)
    targetOutputs, targetTokenIndex, maxTgtSeqLength = vectorizeTargetData(targetTexts, targetCharacters, 1)
    targetIndexToken = createReverseDictionary(targetTokenIndex)
    model, encoderModel, decoderModel = trainModel(inputData, decoderInputs, targetOutputs, maxInSeqLength, maxTgtSeqLength, len(inputCharacters), len(targetCharacters))
    sampleSize = 100
    testOnInDomainDataAndEvaluate(numberOfSamples, sampleSize, inputData, targetTexts, encoderModel, decoderModel, targetCharacters, targetTokenIndex, targetIndexToken, maxTgtSeqLength)
    outOfDomainData = ['1 101 1004 9888', '10005 10006 10007 10008', '11000 12000 13999 17001']
    outOfDomainGoldOutput = ['2 102 1005 9889', '10006 10007 10008 10009', '11001 12001 14000 17002']
    outOfDomainDataArray = vectorizeTestData(outOfDomainData, maxInSeqLength, inputTokenIndex)
    testOnOutDomainDataAndEvaluate(outOfDomainDataArray, outOfDomainGoldOutput, encoderModel, decoderModel, targetCharacters, targetTokenIndex, targetIndexToken, maxTgtSeqLength)


def writeListToFile(filePath, dataList):
    with open(filePath, 'w', encoding='utf-8') as fileWrite:
        fileWrite.write('\n'.join(dataList))


if __name__ == '__main__':
    main()
