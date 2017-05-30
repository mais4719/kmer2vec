Training Results
----------------
100 dimensional embeddings from traning on the human reference genome GCA_000001405.15 GRCh38 (decoy, chrEBV, and HLA were excluded). Allowing kmer sizes between 3 and 8 nucleotides with 15 kmers padding.


IPython Example:
```
In [1]: import numpy as np

In [2]: emb = np.load('max8_min3_mers_15padding_100embedding.npy')

In [3]: emb.shape
Out[3]: (87360, 100)
```

Why 87360x100, considering all kmers between 3 to 8 gives 87,360 possible combinations:
```
In [4]: sum([4**i for i in range(3, 9)])
Out[4]: 87360
```

