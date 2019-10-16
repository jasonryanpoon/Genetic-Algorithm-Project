# available @ https://scikit-learn.org/ 
import sklearn
import sklearn.naive_bayes
import sklearn.decomposition
# available @ https://pandas.pydata.org/ 
import pandas

TrainPercent = 0.90
TestPercent = 0.10
Stats = []

def PCA(df):
    features = df.iloc[0:int(df.shape[0]),0:df.shape[1] - 1].values
    classification = df.iloc[0:int(df.shape[0]),df.shape[1] - 1].values 
    classifier = sklearn.decomposition.PCA()
    dfr = pandas.DataFrame(classifier.fit_transform(features, classification))
    
    features = dfr.iloc[0:int(dfr.shape[0]*TrainPercent),].values
    classification = df.iloc[0:int(df.shape[0]*TrainPercent),df.shape[1] - 1].values 
    
    classifier = sklearn.naive_bayes.GaussianNB()
    classifier.fit(features, classification)
    
    features = dfr.iloc[int(dfr.shape[0]*TestPercent):,].values
    classification = df.iloc[int(df.shape[0]*TestPercent):,df.shape[1] - 1].values
    
    fitness = classifier.score(features, classification)
    Stats.insert(len(Stats), fitness)

if __name__ == '__main__':
    PCA(pandas.read_csv('ccf.csv'))
    pandas.DataFrame(Stats).to_csv('PCA.csv')