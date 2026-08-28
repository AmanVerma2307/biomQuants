import numpy as np
from biomQuant.advancedAcceptance import comp_advancedAcceptance

biomQuant = np.random.normal(size=(11,))
e_prime = np.random.normal(size=(11,))

embeddings = np.random.normal(size=(100,32))
labels = np.random.randint(low=0,high=10,size=(100,))

print(comp_advancedAcceptance(biomQuant,
                                e_prime,
                                embeddings,
                                labels))