import numpy as np
import lda
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()

my_dict = {}
my_dict[0] = [0, 1, 2]
my_dict[1] = [3]
m_set_dict = {2: [0]}
c_set_dict = {3: [0]}

model = lda.LDA(n_topics=20, alpha=0.1, alpha_=0, n_iter=100, tis_dict=my_dict,
                m_set_dict=m_set_dict, c_set_dict=c_set_dict)
model.fit(X)

print model.nzw_[:, 0]
print model.nzw_[:, 1]
print model.nzw_[:, 2]
print model.nzw_[:, 3]


# , m_set_dict=m_set_dict
