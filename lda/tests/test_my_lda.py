import numpy as np
import lda
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()

model = lda.LDA(n_topics=20, alpha=0.1, alpha_=0.01, n_iter=200, random_state=1)
model.fit(X)

print model.nz_
