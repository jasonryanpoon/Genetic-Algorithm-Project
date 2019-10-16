import pandas
# available @ https://matplotlib.org/  
import matplotlib.pyplot

Versions = 4
Generations = 10

def Graph():
    matplotlib.pyplot.figure(0)
    for v in range(Versions):
        matplotlib.pyplot.plot(range(Generations), pandas.read_csv("V"+str(v)+".csv").iloc[:,1].values, label="GAV"+str(v)+".py")
    matplotlib.pyplot.xlabel("Generation")
    matplotlib.pyplot.ylabel("Mean Fitness Level of the Population")
    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()

if __name__ == '__main__':
    Graph()