import numpy as np
import lda
import lda.datasets

X = lda.datasets.load_reuters()
vocab = lda.datasets.load_reuters_vocab()

model = lda.LDA(n_topics=20, alpha=0.1, alpha_=0, n_iter=100, random_state=1, topic_in_set_dict={0: 2})
model.fit(X)

print model.doc_topic_[0:3]
print model.ndz_[0:3]
