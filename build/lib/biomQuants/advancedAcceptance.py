from biomQuants.rankDev import *
from biomQuants.icgd import *
from biomQuants.trendMatch import *
from biomQuants.acceptanceScore import *

def comp_advancedAcceptance(biomQuant,
                            e_prime,
                            embeddings,
                            labels,
                            beta=0.75,
                            nu=1,
                            kappa=1,
                            lambdaVal=2,
                            G=10,
                            normalize=True
                            ):
    
    """
    Function to return normalized or vanilla advanced acceptance score

    INPUTS:-
    1) biomQaunt: Biometric characteristics corresponding to G gestures
    2) e_prime: Ground truth scores for the G gestures
    3) embeddings: Embeddings for ICGD score computation
    4) labels: Gesture labels
    5) beta: Scaling parameter for ICGD score
    6) nu: Scaling parameter for trend match distance
    7) kappa: Scaling parameter for rank deviation
    8) lambdaVal: Scaling parameter for Relevance
    9) G: Total number of gestures
    10) normalize: If True, the nAr* is returned. Default: True

    OUTPUTS:-
    1) Ar*: Advanced acceptance score, normalized version.
    """

    def preProcess(inputVec):
        inputVec = (inputVec - np.mean(inputVec))/np.std(inputVec)
        return inputVec/np.linalg.norm(inputVec)
    
    biomQuant = preProcess(biomQuant)
    e_prime = preProcess(e_prime)

    Ar = compAr(biomQuant,
                e_prime,
                G,
                normalizer=False,
                relevance=False,
                lambda_scale=lambdaVal,
                kappa=kappa) # Acceptance score (relevance + rank deviation)

    icgdScore = compICGD(embeddings,
                         labels) # ICGD score
    
    psi = compTrendMatchDist(biomQuant,
                             e_prime,
                             G) # Trend match distance
    
    if normalize==False:
        return Ar*((np.log2(2+nu*psi))**(-1/2))*np.exp(-beta*icgdScore)
    
    else:
        Ar_max = compAr(biomQuant,
                        e_prime,
                        G,
                        normalizer=True,
                        relevance=False,
                        lambda_scale=lambdaVal,
                        kappa=kappa) # Ar(e_prime)
        
        return (Ar*((np.log2(2+nu*psi))**(-1/2))*np.exp(-beta*icgdScore))/(Ar_max)
    


if __name__ == "__main__":

    biomQuant = np.random.normal(size=(11,))
    e_prime = np.random.normal(size=(11,))

    embeddings = np.random.normal(size=(100,32))
    labels = np.random.randint(low=0,high=10,size=(100,))

    print(comp_advancedAcceptance(biomQuant,
                                  e_prime,
                                  embeddings,
                                  labels))
    

