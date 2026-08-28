import numpy as np
from biomQuants.advancedAcceptance import *
from biomQuants.quantifiers import *
from biomQuants.measures import *


embeddings = np.load('./assets/MS_ViViT_1pt5-1_HandLogin.npz')['arr_0']
labels = np.load('./assets/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
labelIds = np.load('./assets/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
e_prime = np.array([0.23,0.10,0.51,0.02])

scores = getScores('./assets/MS_ViViT_1pt5-1_HandLogin.npz',
                    quantifier='dgbqa',
                    y_dev=labels,
                    y_dev_id=labelIds,
                    G_total=4,
                    I_total=16)
print(scores)


print(comp_advancedAcceptance(scores,
                              e_prime,
                              embeddings,
                              labels))

print(compute_RPP(scores,e_prime,4))