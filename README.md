kmer2vec
========
Unsupervised approach for feature (kmer embeddings) extraction from a provided reference genome. This
code is built on the word2vec model by Mikolov et al. You can find a good overview/tutorial within 
Tensorflow's tutorials [here](https://www.tensorflow.org/tutorials/word2vec).

#### Reference Vocabulary hg38
Folder ```reference_vocabulary``` contains code and a make file for downloading and pre-process the 
human reference genome GCA_000001405.15 GRCh38.

This will download the reference genome and repeatmasker, and create frequency of 6-mers in SINE, LINE, and ALL:
```
# make
```
Other k-mer sizes can be computed by running:
```
# make KMER_SIZE=8
```

#### Train on Reference
To start training on a reference genome (fasta file with corresponding faidx) run for example:
```
# ./kmer2vec.py \
    --fa_file reference_vocabulary/GRCh38_full_hs38d1_decoy.fa \
    --validation_vocabulary reference_vocabulary/all_6-mers.tsv \
    --min_kmer_size 3 \
    --max_kmer_size 8 \
    --padding 10 \
    --learning_rate 0.8 \
    --embedding_size 100 \
    --num_neg_samples 100
```

For debuging you can use an interactive session by adding the flagg ```--interactive```. This 
will make the program jump into an ipython shell after execution.

#### Notebooks
This is a work in progress. Notebooks are used to explore and find new ideas to improve the process.

##### Vector Cosine vs Nedleman-Wunsch Score
Looks at the correlation between cosine similarity and Nedleman-Wunsch score.

##### visualize_kmer_word2vec
Notebook visualize embeddings using t-SNEs and pre-computed SINE and LINE k-mer 
frequencies (created by make in folder "reference_vocabulary")

