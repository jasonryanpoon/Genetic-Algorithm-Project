import random
import statistics
# available @ https://scikit-learn.org/ 
import sklearn
import sklearn.naive_bayes
# available @ https://pandas.pydata.org/ 
import pandas

# V1 - A higher mutation rate of 99% instead of 1%

MutationPercent = 0.99
InitialPopulationPercent = 0.0001
TrainPercent = 0.90
TestPercent = 0.10
Generations = 10
Stats = []

def InitialPopulation(rows, cols): 
    pop = [[False, False, True, True, False, False, True, True, True, False, False, True, True, True, False, False, True, True, True, False, True, True, True, True, True, False, True, False, True, False, False], 
    [False, True, False, True, False, True, True, False, True, True, True, False, False, True, False, True, False, False, False, False, False, True, False, True, False, True, False, False, False, False, True], 
    [True, True, True, True, True, False, True, True, False, True, False, True, False, True, False, False, False, True, True, False, False, False, False, False, False, True, True, True, True, True, False], 
    [False, True, True, True, False, False, True, False, False, False, False, True, False, False, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, False], 
    [True, True, False, False, False, False, True, True, True, True, False, True, False, False, True, False, True, True, True, False, True, True, False, True, True, False, False, True, False, True, True], 
    [False, True, True, True, False, False, False, True, False, False, True, False, False, False, False, False, True, False, False, False, True, True, False, False, False, False, True, False, True, False, True], 
    [False, True, True, False, True, False, True, False, False, True, True, False, False, True, True, False, False, False, True, True, True, True, False, True, True, False, False, False, False, False, False], 
    [True, True, False, True, False, True, False, False, True, True, False, True, False, False, True, False, True, False, True, False, False, False, True, True, True, True, False, True, False, False, False], 
    [False, True, True, True, True, True, False, True, True, True, True, False, False, False, False, False, True, False, True, False, False, False, False, False, False, False, True, True, True, True, True], 
    [True, True, False, False, True, True, True, False, True, True, False, False, False, True, False, True, False, False, True, True, False, True, False, False, False, True, False, True, False, True, False], 
    [False, True, True, False, True, False, False, True, False, False, False, True, True, False, True, False, True, True, False, False, False, True, False, True, False, True, False, False, True, True, True], 
    [False, False, True, True, False, True, False, False, False, True, True, False, True, False, False, True, True, True, True, True, False, True, True, True, True, False, True, True, True, True, True], 
    [False, False, True, False, True, True, True, False, False, True, False, False, True, False, False, False, True, True, False, False, True, True, False, True, True, True, True, False, True, False, False], 
    [True, True, False, True, False, False, False, True, False, False, True, False, True, True, False, False, False, True, False, True, False, False, True, False, False, True, False, True, False, True, True], 
    [False, False, True, False, False, True, False, False, False, False, True, False, True, False, False, True, False, True, True, False, False, True, True, True, True, True, False, False, True, True, True], 
    [False, False, True, False, False, True, False, True, False, True, False, False, False, True, True, True, False, True, False, True, False, False, False, True, False, True, False, True, True, True, False], 
    [False, True, False, True, False, False, False, False, False, False, True, True, False, True, False, True, False, False, False, True, True, False, True, True, False, False, True, False, False, False, False], 
    [False, False, True, False, True, True, True, False, False, False, True, False, True, False, False, False, False, False, False, True, False, False, True, False, False, True, False, True, True, True, False], 
    [False, False, True, True, False, False, True, True, False, False, False, False, True, True, True, True, False, False, False, False, False, False, True, False, False, True, True, False, True, True, True], 
    [False, False, False, False, False, False, False, True, True, False, True, False, False, True, True, True, False, False, False, False, True, False, True, False, False, True, False, False, True, True, True], 
    [False, True, False, True, False, False, False, False, True, False, False, True, True, True, False, False, True, False, True, True, True, False, False, False, False, False, False, False, False, False, False], 
    [False, True, True, True, True, True, True, True, False, False, True, True, False, True, True, False, True, False, False, True, True, True, True, False, True, False, False, False, False, False, False], 
    [False, False, True, False, False, True, False, False, True, True, True, False, False, False, False, True, True, True, False, True, False, False, False, False, True, False, True, True, True, False, True], 
    [False, True, False, True, False, False, True, True, False, False, True, False, True, True, True, True, False, False, False, True, True, False, False, False, False, True, True, True, True, False, False], 
    [False, False, False, True, True, False, False, False, True, True, True, True, False, True, True, True, False, True, False, False, False, True, True, True, False, True, False, True, False, False, False], 
    [False, False, False, True, True, False, True, True, False, True, False, True, False, False, True, False, True, True, True, False, True, False, True, True, False, False, True, False, True, False, False], 
    [False, True, True, True, True, True, False, False, False, True, False, True, True, False, False, False, False, False, True, False, False, False, False, False, True, False, True, True, False, True, True], 
    [True, False, True, False, False, False, False, False, True, False, False, True, False, True, True, True, True, False, True, False, True, True, True, True, True, True, False, True, False, True, True]]
    # pop = []
    # for i in range(int(rows * InitialPopulationPercent)):
    #     individual = [False] * cols
    #     for gene in range(cols):
    #         if(random.random() < 0.5):
    #             individual[gene] = True
    #     pop.insert(len(pop), individual)
    return pop

# based on https://stackoverflow.com/a/37199623
def Normalize(index):
    df = pandas.DataFrame(index)
    return sklearn.preprocessing.MinMaxScaler().fit_transform(df)

def Fitness(df, pop):
    index = []
    for individual in pop:
        dfc = df.copy(deep=True)
        offset = 0
        for gene in range(dfc.shape[1] - 1):
            if(not individual[gene]):
                dfc.drop(dfc.columns[[gene - offset]], axis=1, inplace=True)
                offset = offset + 1
        features = dfc.iloc[0:int(dfc.shape[0]*TrainPercent),0:dfc.shape[1] - 1].values
        classification = dfc.iloc[0:int(dfc.shape[0]*TrainPercent),dfc.shape[1] - 1].values 
        classifier = sklearn.naive_bayes.GaussianNB()
        classifier.fit(features, classification)
        features = dfc.iloc[int(dfc.shape[0]*TestPercent):,0:dfc.shape[1] - 1].values
        classification = dfc.iloc[int(dfc.shape[0]*TestPercent):,dfc.shape[1] - 1].values
        fitness = classifier.score(features, classification)
        index.insert(len(index), fitness)
    Stats.insert(len(index), statistics.mean(index))
    return Normalize(index)

def Selection(pop, index):
    while(True):
        individual = random.randrange(len(pop))
        if(random.random() < index[individual]):
            break
    return individual

def Crossover(x, y):
    split = random.randrange(len(x))
    child = x[:split] + y[split:]
    return child

def Mutate(child):
    child[random.randrange(len(child))] = not child[random.randrange(len(child))]
    return child

# based on Artificial Intelligence: A Modern Approach pseudocode on page 129
def GA(df):
    pop = InitialPopulation(df.shape[0], df.shape[1])
    for g in range(Generations):
        new = []
        index = Fitness(df, pop)
        for i in range(len(pop)):
            x = Selection(pop, index)
            y = Selection(pop, index)
            child = Crossover(pop[x], pop[y])
            if(random.random() < MutationPercent):
                child = Mutate(child)
            new.insert(len(new), child)
        pop = new

if __name__ == '__main__':
    GA(pandas.read_csv('ccf.csv'))
    pandas.DataFrame(Stats).to_csv('V1.csv')