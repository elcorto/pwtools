
def nweights_mlgraph(netarch, biases=False):
    """Number of weights for NNs from ffnet.mlgraph().

    Parameters
    ----------
    netarch : tuple, e.g. (1,5,5,1)
    biases : bool

    Examples
    --------
    >>> netarch=(1,5,1)
    >>> net=ffnet.ffnet(ffnet.mlgraph(netarch, biases=True))
    >>> net
    Feed-forward neural network: 
    inputs:     1 
    hiddens:    5 
    outputs:    1 
    connections and biases:   16
    >>> len(net.weights)
    16
    >>> net.graph.number_of_edges()
    16
    >>> nweights_mlgraph(netarch, biases=True)
    16
    """
    ln = len(netarch)
    nw = sum([netarch[ii]*netarch[ii+1] for ii in range(ln-1)])
    if biases:
        nw += (sum(netarch[1:-1]) + netarch[-1])
    return nw        
