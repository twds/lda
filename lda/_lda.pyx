#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

from cython.operator cimport preincrement as inc, predecrement as dec
from libc.stdlib cimport malloc, free
from libc.math cimport log


cdef extern from "gamma.h":
    cdef double lda_lgamma(double x) nogil


cdef double lgamma(double x) nogil:
    if x <= 0:
        with gil:
            raise ValueError("x must be strictly positive")
    return lda_lgamma(x)

cdef double max(double a, double b) nogil:
    if a < b :
        return b
    else:
        return a

cdef int searchsorted(double* arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin


def _sample_topics(int[:] WS, int[:] DS, int[:] ZS, int[:, :] nzw, int[:, :] ndz, int[:] nz, double[:, :] tis_dict,
                   int[:, :] m_set_dict, int[:, :] c_set_dict, double wr_lambda,
                   double alpha, double eta, double alpha_, double term_num, double[:] rands):
    cdef int i, k, w, d, z, z_new
    cdef double r, dist_cum
    cdef int N = WS.shape[0]
    cdef int n_rand = rands.shape[0]
    cdef int n_topics = nz.shape[0]
    cdef double eta_sum = 0
    cdef double ratio = 1
    cdef double prior_score = 1
    cdef int m_set_col = m_set_dict.shape[1]
    cdef int c_set_col = c_set_dict.shape[1]
    cdef double* dist_sum = <double*> malloc(n_topics * sizeof(double))
    if dist_sum is NULL:
        raise MemoryError("Could not allocate memory during sampling.")
    with nogil:
        eta_sum = eta * N

        for i in range(N):
            w = WS[i]
            d = DS[i]
            z = ZS[i]

            dec(nzw[z, w])
            dec(ndz[d, z])
            dec(nz[z])

            dist_cum = 0
            for k in range(n_topics):
                if alpha_ != 0:
                    ratio = (nz[k] + alpha_)/(term_num + alpha_ * n_topics)
                    ratio = ratio * n_topics
                prior_score = 1
                for m in range(m_set_col):
                    if m_set_dict[w][m] < 0:
                        break;
                    prior_score += log(max(wr_lambda, nzw[k,m_set_dict[w][m]]))
                    #with gil:
                        #print(nzw[k,m_set_dict[w][m]])
                        #print(k, prior_score)
                for c in range(c_set_col):
                    if c_set_dict[w][m] < 0:
                        break;
                    prior_score += log(1/max(wr_lambda, nzw[k,c_set_dict[w][m]]))
                # eta is a double so cdivision yields a double
                dist_cum += prior_score * tis_dict[w, k] * (nzw[k, w] + eta) / (nz[k] + eta_sum) * (ndz[d, k] + alpha * ratio)
                dist_sum[k] = dist_cum

            r = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = searchsorted(dist_sum, n_topics, r)
            ZS[i] = z_new
            inc(nzw[z_new, w])
            inc(ndz[d, z_new])
            inc(nz[z_new])

        free(dist_sum)


cpdef double _loglikelihood(int[:, :] nzw, int[:, :] ndz, int[:] nz, int[:] nd, double alpha, double[:] alpha_, double eta) nogil:
    cdef int k, d
    cdef int D = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int vocab_size = nzw.shape[1]

    cdef double ll = 0

    # calculate log p(w|z)
    cdef double lgamma_eta, lgamma_alpha
    with nogil:
        lgamma_eta = lgamma(eta)

        ll += n_topics * lgamma(eta * vocab_size)
        for k in range(n_topics):
            ll -= lgamma(eta * vocab_size + nz[k])
            for w in range(vocab_size):
                # if nzw[k, w] == 0 addition and subtraction cancel out
                if nzw[k, w] > 0:
                    ll += lgamma(eta + nzw[k, w]) - lgamma_eta

        # calculate log p(z)
        for d in range(D):
            ll += (lgamma(alpha * n_topics) -
                    lgamma(alpha * n_topics + nd[d]))
            for k in range(n_topics):
                if ndz[d, k] > 0:
                    ll += lgamma(alpha_[k] + ndz[d, k]) - lgamma(alpha_[k])
        return ll
