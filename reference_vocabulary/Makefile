URL38:="ftp://ftp.ncbi.nlm.nih.gov/genomes/genbank/vertebrate_mammalian/Homo_sapiens/all_assembly_versions/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_full_plus_hs38d1_analysis_set.fna.gz"

KMER_SIZE:=6
REF_GENOME:=GRCh38_full_hs38d1_decoy.fa

NPROCS:=1
OS:=$(shell uname -s)

ifeq ($(OS),Linux)
	NPROCS:=$(shell grep -c ^processor /proc/cpuinfo)
endif
ifeq ($(OS),Darwin) # Assume Mac OS X
	NPROCS:=$(shell system_profiler | awk '/Number Of CPUs/{print $4}{next;}')
endif

all: rmsk_LINE_$(KMER_SIZE)-mers.tsv  rmsk_SINE_$(KMER_SIZE)-mers.tsv all_$(KMER_SIZE)-mers.tsv

refgenome: GRCh38_full_hs38d1_decoy.fa.fai GRCh38_full_hs38d1_decoy.fa

rmsk.txt.gz :
	rsync -avzP rsync://hgdownload.cse.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz .

rmsk_LINE.bed : rmsk.txt.gz
	zcat rmsk.txt.gz | awk -v class=LINE 'BEGIN{OFS="\t"}{if($$12==class) {print($$6, $$7, $$8, $$11 ":" $$12 ":" $$13);}}' > rmsk_LINE.bed

rmsk_SINE.bed : rmsk.txt.gz
	zcat rmsk.txt.gz | awk -v class=SINE 'BEGIN{OFS="\t"}{if($$12==class) {print($$6, $$7, $$8, $$11 ":" $$12 ":" $$13);}}' > rmsk_SINE.bed

rmsk_LINE_$(KMER_SIZE)-mers.tsv : rmsk_LINE.bed ../reference_vocabulary.py $(REF_GENOME) $(REF_GENOME).fai
	../reference_vocabulary.py --output rmsk_LINE_$(KMER_SIZE)-mers.tsv -f "decoy" -f "chrEBV" -f "HLA" -f "alt" --bed_file rmsk_LINE.bed -w $(KMER_SIZE) -p $(NPROCS) $(REF_GENOME)

rmsk_SINE_$(KMER_SIZE)-mers.tsv : rmsk_SINE.bed ../reference_vocabulary.py $(REF_GENOME) $(REF_GENOME).fai
	../reference_vocabulary.py --output rmsk_SINE_$(KMER_SIZE)-mers.tsv -f "decoy" -f "chrEBV" -f "HLA" -f "alt" --bed_file rmsk_SINE.bed -w $(KMER_SIZE) -p $(NPROCS) $(REF_GENOME)

all_$(KMER_SIZE)-mers.tsv : ../reference_vocabulary.py $(REF_GENOME) $(REF_GENOME).fai
	../reference_vocabulary.py --output all_$(KMER_SIZE)-mers.tsv -f "decoy" -f "chrEBV" -f "HLA" -f "alt" -w $(KMER_SIZE) -p $(NPROCS) $(REF_GENOME)

GRCh38_full_hs38d1_decoy.fa :
	(wget -O- $(URL38) | gzip -dc) > GRCh38_full_hs38d1_decoy.fa

GRCh38_full_hs38d1_decoy.fa.fai : GRCh38_full_hs38d1_decoy.fa
	samtools faidx GRCh38_full_hs38d1_decoy.fa

clean:
	rm rmsk.txt.gz
	rm rmsk_*.bed
	rm rmsk_*_$(KMER_SIZE)-mers.tsv
	rm all_$(KMER_SIZE)-mers.tsv
	rm GRCh38_full_hs38d1_decoy.*

