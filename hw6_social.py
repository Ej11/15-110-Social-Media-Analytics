"""
15-110 Hw6 - Social Media Analytics Project
Name: E.j. Ezuma-Ngwu
AndrewID: ufe
"""

import hw6_social_tests as test

project = "Social" # don't edit this

### WEEK 1 ###

import pandas as pd
import nltk
nltk.download('vader_lexicon', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
endChars = [ " ", "\n", "#", ".", ",", "?", "!", ":", ";", ")" ]

'''
makeDataFrame(filename)
#3 [Check6-1]
Parameters: str
Returns: dataframe
'''
def makeDataFrame(filename):
    df = pd.read_csv(filename)
    return df


'''
parseName(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseName(fromString):
    first = fromString.find(":") +2
    last = fromString.find("(") -1
    name = fromString[first:last]
    return name


'''
parsePosition(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parsePosition(fromString):
    first = fromString.find("(")+1
    last = fromString.find("from")-1
    position = fromString[first:last]
    return position


'''
parseState(fromString)
#4 [Check6-1]
Parameters: str
Returns: str
'''
def parseState(fromString):
    first = fromString.find("from")+ 5
    #(would normally be +2 for a singular string but because it is a word you do +5 to account for "rom" in "from"
    last = fromString.find(")")
    state = fromString[first:last]
    return state


'''
findHashtags(message)
#5 [Check6-1]
Parameters: str
Returns: list of strs
'''
def findHashtags(message):
    result = []
    for i in range(len(message)):
        hash = ""
        #message[i] is the letter and i is the index
        if message[i] == "#":
            j= i+1
            while j < len(message) and message[j] not in endChars:
                j= j+1
            single = message[i:j]
            result.append(single)
    return result



'''
getRegionFromState(stateDf, state)
#6 [Check6-1]
Parameters: dataframe ; str
Returns: str
'''
def getRegionFromState(stateDf, state):
    sub1 = stateDf[stateDf["state"]==state]
    region = sub1.iloc[0]["region"]
    return region


'''
addColumns(data, stateDf)
#7 [Check6-1] & #2 [Check6-2]
Parameters: dataframe ; dataframe
Returns: None
'''
def addColumns(data, stateDf):
    names=[]
    positions=[]
    states=[]
    regions=[]
    hashtags=[]
    sentiment = []
    sent = SentimentIntensityAnalyzer()
    for label in data["label"]:
        names.append(parseName(label))
        positions.append(parsePosition(label))
        states.append(parseState(label))
        regions.append(getRegionFromState(stateDf, parseState(label)))
    for text in data["text"]:
        hashtags.append(findHashtags(text))
        sentiment.append(findSentiment(sent,text))
    data["name"]= names
    data["position"] = positions
    data["state"] = states
    data["region"] = regions
    data["hashtags"] = hashtags
    data["sentiment"] = sentiment





### WEEK 2 ###

'''
findSentiment(classifier, message)
#1 [Check6-2]
Parameters: SentimentIntensityAnalyzer ; str
Returns: str
'''
def findSentiment(classifier, message):
    score = classifier.polarity_scores(message)['compound']
    if score < -0.1:
        return "negative"
    if score > 0.1:
        return "positive"
    else:
        return "neutral"

'''
getDataCountByState(data, colName, dataToCount)
#3 [Check6-2]
Parameters: dataframe ; str ; str
Returns: dict mapping strs to ints
'''
def getDataCountByState(data, colName, dataToCount):
    if colName != "":
        data = data[data[colName] == dataToCount]
    sentOfState= { }
    for states in data["state"]:
        if states in sentOfState:
            sentOfState[states] += 1
        elif states not in sentOfState:
            sentOfState[states] = 1
    return sentOfState


'''
getDataForRegion(data, colName)
#4 [Check6-2]
Parameters: dataframe ; str
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def getDataForRegion(data, colName):
    d = {}
    for index, row in data.iterrows():
        message = row[colName]
        region = row["region"]
        if region not in d:
            d[region] = {}
        if message not in d[region]:
            d[region][message] = 0
        d[region][message] += 1

    return d
    #only need to do one for loop and check if the region is in the dictionary and if not add it, else add the message of the region



'''
getHashtagRates(data)
#5 [Check6-2]
Parameters: dataframe
Returns: dict mapping strs to ints
'''
def getHashtagRates(data):

    hashRates= {}
    for hashtagCount in data["hashtags"]:
        for tags in hashtagCount:
            if tags in hashRates:
                hashRates[tags] += 1
            elif tags not in hashRates:
                hashRates[tags] = 1

    return hashRates



#REFER TO TEST CASES NOT CSV

'''
mostCommonHashtags(hashtags, count)
#6 [Check6-2]
Parameters: dict mapping strs to ints ; int
Returns: dict mapping strs to ints
'''
def mostCommonHashtags(hashtags, count):
    commonHash ={}

    while len(commonHash) < count:
        highestNum = 0
        highestKey = None
        for key in hashtags:
            if key not in commonHash:
                if hashtags[key] > highestNum:
                    highestNum = hashtags[key]
                    highestKey = key
        commonHash[highestKey] = highestNum
    return commonHash





'''
getHashtagSentiment(data, hashtag)
#7 [Check6-2]
Parameters: dataframe ; str
Returns: float
'''
import statistics
def getHashtagSentiment(data, hashtag):
    result = 0
    avgCount = 0
    count = []                          #FIX THIS BEFORE SUBMISSION
    for index, row in data.iterrows():
        hashtagsRow = row["sentiment"]
        if hashtag in row["hashtags"]:
            if hashtagsRow == "positive":
                count.append(1)
            if hashtagsRow == "negative":
                count.append(-1)
            if hashtagsRow == "neutral":
                count.append(0)
    allsum = sum(count)
    numOfHash = int(len(count))
    avgCount = allsum/numOfHash
    result = avgCount
    return result

#FIX FIX




### WEEK 3 ###

'''
graphStateCounts(stateCounts, title)
#2 [Hw6]
Parameters: dict mapping strs to ints ; str
Returns: None
'''
def graphStateCounts(stateCounts, title):
    import matplotlib.pyplot as plt
    labels = []
    yVals = []
    for state in stateCounts:
        labels.append(state)
        yVals.append(stateCounts[state])

    plt.bar(labels, yVals, color="green")
    plt.title(title)
    plt.xticks(rotation ='vertical')
    plt.xlabel("States")
    plt.ylabel("Number")
    plt.show()
    return

'''
graphTopNStates(stateCounts, stateFeatureCounts, n, title)
#3 [Hw6]
Parameters: dict mapping strs to ints ; dict mapping strs to ints ; int ; str
Returns: None
'''
def graphTopNStates(stateCounts, stateFeatureCounts, n, title):
    topN ={}
    while len(topN) < n:
        total = 0
        fTotal = 0
        bestFRate = -1
        highestState = None
        for state in stateFeatureCounts:
            total = (stateCounts[state])
            fTotal = (stateFeatureCounts[state])
            tempRate = fTotal/total
            if state not in topN and tempRate > bestFRate:
                bestFRate = tempRate
                highestState = state
        topN[highestState] = bestFRate

    graphStateCounts(topN, title)
    return


'''
graphRegionComparison(regionDicts, title)
#4 [Hw6]
Parameters: dict mapping strs to (dicts mapping strs to ints) ; str
Returns: None
'''
def graphRegionComparison(regionDicts, title):
    featureNames=[]
    regionNames=[]
    regFeaLists=[]
    for regions in regionDicts:
        if regions not in regionNames:
            regionNames.append(regions)
        for features in regionDicts[regions]:
            if features not in featureNames:
                featureNames.append(features)

    for regions in regionNames:
        tempFeatLists = []
        for features in featureNames:
            if features in regionDicts[regions]:
                tempFeatLists.append(regionDicts[regions][features])
        regFeaLists.append(tempFeatLists)

    sideBySideBarPlots(featureNames,
regionNames, regFeaLists, title)
    return


'''
graphHashtagSentimentByFrequency(data)
#4 [Hw6]
Parameters: dataframe
Returns: None
'''
def graphHashtagSentimentByFrequency(data):
    d = getHashtagRates(data)
    mch = mostCommonHashtags(d, 50)
    hashtags=[]
    freqs=[]
    sentiScores=[]
    for htgs in mch:
        hashtags.append(htgs)
        freqs.append(mch[htgs])
        sentiScores.append(getHashtagSentiment(data, htgs))
    scatterPlot(freqs, sentiScores, hashtags, "Hashtag Sentiment By Frequency in Messages")
    return


#### WEEK 3 PROVIDED CODE ####
"""
Expects 3 lists - one of x labels, one of data labels, and one of data values - and a title.
You can use it to graph any number of datasets side-by-side to compare and contrast.
"""
def sideBySideBarPlots(xLabels, labelList, valueLists, title):
    import matplotlib.pyplot as plt

    w = 0.8 / len(labelList)  # the width of the bars
    xPositions = []
    for dataset in range(len(labelList)):
        xValues = []
        for i in range(len(xLabels)):
            xValues.append(i - 0.4 + w * (dataset + 0.5))
        xPositions.append(xValues)

    for index in range(len(valueLists)):
        plt.bar(xPositions[index], valueLists[index], width=w, label=labelList[index])

    plt.xticks(ticks=list(range(len(xLabels))), labels=xLabels, rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Expects that the y axis will be from -1 to 1. If you want a different y axis, change plt.ylim
"""
def scatterPlot(xValues, yValues, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xValues, yValues)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xValues[i], yValues[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.ylim(-1, 1)

    # a bit of advanced code to draw a line on y=0
    ax.plot([0, 1], [0.5, 0.5], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()