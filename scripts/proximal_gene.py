import sys
import pandas as pd
import scvi as sc



'''

    This function helps to associate gene_ids to peaks in ATAC-seq data
    using GFF annotations
    Assumption: peak effects genes that satisfy
    peak_mean + 5000 bp > gene_start > peak_mean - 5000 bp


    peaks: ATAC-seq AnnData
    gff:   pandas.DataFrame containing Gene annotations in GFF file  


    gff_file_name = "/home/pcddas/expvae/Homo_sapiens.GRCh38.104.chr.gff3.gz"

    # read csv function with arguments to parse the file
   9yy gff = pd.read_csv(
        gff_file_name,
        sep='\t',
        header=None,
        comment="#",
        names=("seqname", "source", "feature", "start",
               "end", "score", "strand", "frame", "attribute"),
    )

    # return the dataframe
    gff.head()

'''

def proximal_gene(peaks, gff):

    GFF_GENES = gff[gff["feature"]=="gene"].copy()

    GFF_GENES["gene_ids"] = [(a.split(";")[0]).replace("ID=gene:","") for i, a in enumerate(GFF_GENES["attribute"].values) ]

    GFF_GENES["seqname"] = [str(v) for i, v in enumerate(GFF_GENES["seqname"].tolist())]

    peaks.var["seqname"] = [(v.split(":")[0]).replace("chr","") for i,v in enumerate(peaks.var.index.tolist())]

    peaks.var["peak_start"] = [int((v.split(":")[1]).split("-")[0]) for i,v in enumerate(peaks.var.index.tolist())]
    peaks.var["peak_end"] = [int((v.split(":")[1]).split("-")[-1]) for i,v in enumerate(peaks.var.index.tolist())]

    peaks.var["peak_mean"] = 0.5*(peaks.var["peak_start"] + peaks.var["peak_end"])

    peaks.var = peaks.var.astype({"peak_mean": 'int64'})

    peaks.var["proximal_gene"] = "None"

    peaks.var.sort_values(by = ["seqname","peak_start"], inplace = True)

    for i, m in enumerate(peaks.var["peak_mean"].tolist()):
        proximal_peaks = GFF_GENES[GFF_GENES["start"].between(m-5000,m+5000)].copy()
        proximal_gene_data = proximal_peaks[proximal_peaks["seqname"] == peaks.var["seqname"].values[i]]
        if not proximal_gene_data.empty:
            proximal_gene = proximal_gene_data["gene_ids"].values[0]
            peaks.var["proximal_gene"].values[i] = proximal_gene
            peaks.var["proximal_gene_start"].values[i] = proximal_gene_data["start"].values[0]
            
    
    return peaks

#if __name__ == "__main__":
#    peaks_file = sys.argv[1]
#    gff_file = sys.argv[2]
#    
#    peaks = sc.read_h5ad(peaks_file)
#
#    # read csv function with arguments to parse the file
#    gff = pd.read_csv(
#        gff_file,
#        sep='\t',
#        header=None,
#        comment="#",
#        names=("seqname", "source", "feature", "start",
#               "end", "score", "strand", "frame", "attribute"),
#    )
#
#    proximal_gene(peaks,gff)
