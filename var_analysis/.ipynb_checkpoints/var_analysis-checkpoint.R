# Analyze withing and between corpus coefficient of variation. 
# Given many corpora, consider word-frequency of many document-corpus.  
# Use an exact binomial test for each word, and record the rank of
# the P-value associated with that word. In addition, record the 
# coeficient of variation of the word across the corpus. We reveal
# that words whose P-value is small also have low coefficient of 
# variation. 


# Parameters
# ==========
VOCAB_SIZE = 2000    # only consider most common words
MIN_CNT = 0
ANSCOMB_PARAM = 3/8  # additive parameter in variance-stabilizing transformation
N_SAMPLES = 990      # number of document-corpus samples
TOP_RANK = 100       # only record TOP_RANK ranks
ALPHA = .25          # HC limit parameter
M = 4                # group M documents 



# Suppress summarise info
library(dplyr, warn.conflicts = FALSE)
options(dplyr.summarise.inform = FALSE)


library("TableHC")
library(tidyverse)
library(tidytext)
library(tidyverse)


source("two_dtc_stats.R")
source("load_data.R")


printf <- function(...)print(sprintf(...))
data = read_data_dtc("./doc_term_counts_PAN2019.csv")


# most common 2000 words
common.words <- data %>% 
              group_by(lemma) %>% 
              summarise(n = sum(n)) %>%
              arrange(-n) %>%
              head(VOCAB_SIZE)


# only keep most common words
data_red <- data %>% right_join(common.words %>% select(lemma))

#add anscombe to dtc.
anscombe <- data_red %>% mutate(term = lemma) %>%
            group_by(doc_id) %>%
            mutate(T.in_doc = sum(n),
                   r = 2*sqrt(n + ANSCOMB_PARAM) / sqrt(T.in_doc)) %>%
            ungroup() %>% select(-lemma)


# store results in these data frames:
df1 = data_frame()
df0 = data_frame()

lo_classes = unique(anscombe$class)
df_smp <- expand.grid(lo_classes, lo_classes) %>% 
          filter(as.vector(Var2) > as.vector(Var1)) %>%
          sample_n(N_SAMPLES)

pb <- txtProgressBar(min = 0, max = N_SAMPLES, style = 3)

printf("Evaluating coefficinet of variation and binomial P-value rank.")
printf("Going over a pool of %d document-corpus pairs", N_SAMPLES)

for (i in seq(N_SAMPLES)) {
    setTxtProgressBar(pb, i)
    smp = df_smp[i,]
    
    dtc1 = anscombe %>% filter(class == smp$Var1)
    dtc2 = anscombe %>% filter(class == smp$Var2)

    lo_doc_id = unique(dtc1$doc_id)
    lo_doc_id2 = unique(dtc2$doc_id)
    
    for (j in seq(lo_doc_id[1:2])) {
        #dc = lo_doc_id[j]
        k1 = floor(length(lo_doc_id)/M)
        dc = sample(lo_doc_id, k1)
        k2 = floor(length(lo_doc_id2)/M)
        dc2 = sample(lo_doc_id2, k2)

        dtc_corpus = dtc1 %>% filter(!(doc_id %in% dc) )
        dtc_doc = dtc1 %>% filter(doc_id %in% dc)
        dtc_other_corpus = dtc2 %>% filter(!(doc_id %in% dc2) )

        stat_within <- two_dtc_stats(dtc_doc, dtc_corpus, ANSCOMB_PARAM, MIN_CNT)
        stat_between <- two_dtc_stats(dtc_doc, dtc_other_corpus, ANSCOMB_PARAM, MIN_CNT)

        res0 <- stat_within %>%
                filter(rank < TOP_RANK, n.y > 1) %>%
                arrange(abs(p)) %>%
                mutate(rank = order(p),
                       type = 'within',
                       nstd = std.y / mu.y, #coefficient of variation
                       class = paste(smp$Var1,smp$Var2)) %>%
                select(rank,p,type,nstd,
                       z = z2, z_p = z_p,
                       std = std.y,
                       mu = mu.y,
                       class,
                       evar = evar.y) 
        
        p_th = HC.vals(res0$p, stbl = TRUE, alpha = ALPHA)$p.star
        n_th = sum(res0$p <= p_th)
        res0 <- res0 %>% mutate(n_th = n_th)

        res1 <- stat_between %>%
                filter(rank < TOP_RANK, n.y > 1) %>%
                arrange(abs(p)) %>%
                mutate(rank = order(p),
                       type = 'between',
                       nstd = std.y / mu.y, #coefficient of variation
                      class = paste(smp$Var1,smp$Var2)) %>%
                select(rank, type, p, nstd,
                       z = z2,
                       z_p = z_p,
                       std = std.y,
                       mu = mu.y,
                      class,
                       evar = evar.y) 

        p_th = HC.vals(res1$p, stbl = TRUE, alpha = ALPHA)$p.star
        n_th = sum(res1$p <= p_th)
        
        res1 <- res1 %>% mutate(n_th = n_th)
        
        df1 <- rbind(df1, res1)
        df0 <- rbind(df0, res0)
    }
}
close(pb)

write.csv(df0, paste("./df0_",dataset_name,".csv", sep = ""))
write.csv(df1, paste("./df1_",dataset_name,".csv", sep = ""))

