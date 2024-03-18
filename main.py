
from MVC import MVC
import time
KThreshold = 0
K = 500
sampleNum = 10
iterNum = 20
wordsInTopicNum = 20

alpha = 0.1
beta = 0.1
sigma = 0.0004 # sigma^2
kappa = 0.5
_lambda = 0.8




dataset = "News-T-SIMCSE"
 


dataDir = "data/"
outputPath = "result/"


def runMVC(K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir):
    mvc = MVC(K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    mvc.getDocuments()
    for sampleNo in range(1, sampleNum + 1):
        print("SampleNo:" + str(sampleNo))
        mvc.runMVC(sampleNo, outputPath)

if __name__ == '__main__':
    outf = open("time_MVC", "a")
    time1 = time.time()
    runMVC(K, alpha, beta, sigma, kappa, _lambda, iterNum, sampleNum, dataset, wordsInTopicNum, dataDir)
    time2 = time.time()
    outf.write(str(dataset) + "K" + str(K) + "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3)) +
        "sigma" + str(round(sigma, 3)) + "kappa" + str(round(kappa, 3)) + "_lambda" + str(_lambda) +
        "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + "\ttime:" + str(time2 - time1) + "\n")
