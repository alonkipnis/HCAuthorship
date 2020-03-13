#auxiliary functions
library(readr)
library(dplyr)
library(tm)
library(tidytext)
library(tidyverse)
library(data.table)
library(TableHC)

#"~/Google Drive//Data/PAN2018_English/doc_term_counts_PAN2018_test.csv"
path_to_data = "./doc_term_counts_PAN2019_train.csv"

#PAN in doc_term_count format
dtc_raw = read.csv(path_to_data)

dataset_name = "PAN2019_DTC"

dtc_red <- dtc_raw %>%
           filter(author != '<UNK>') %>%
           filter(POS != 'PROPN') %>%
           filter(POS != 'PUNCT') %>%
           filter(POS != 'NUM') %>%
           filter(POS != 'PRON') %>%
           select(dataset, author, doc_id, lemma, n) %>%
           group_by(dataset, author, doc_id, lemma) %>%
           summarise(n = sum(n))  

dtc_red$dataset <- gsub("problem", "p", dtc_red$dataset)
dtc_red$dataset <- gsub("000", "", dtc_red$dataset)

dtc_red$author <- gsub("candidate000", "c", dtc_red$author)
dtc_red$doc_id <- gsub("problem000","p", dtc_red$doc_id)
dtc_red$doc_id <- gsub("candidate000","c", dtc_red$doc_id)
dtc_red$class <- paste(dtc_red$dataset, dtc_red$author, sep = '-')
dtc_red$class <- gsub('PAN-', "", dtc_red$class)

doc.id.data <- dtc_red %>% 
              select(dataset, author, doc_id) %>%
              unique() 

common.words <- dtc_red %>% 
              group_by(lemma) %>% 
              summarise(n = sum(n)) %>%
              arrange(-n) %>%
              head(2000)

#add anscombe to dtc.
dtc_red_red <- dtc_red %>% right_join(common.words %>% select(lemma))

anscombe <- dtc_red_red %>% mutate(term = lemma) %>%
            group_by(doc_id) %>%
            mutate(T.in_doc = sum(n),
                   r = 2*sqrt(n + 2/8) / sqrt(T.in_doc)) %>%
            ungroup() %>% select(-lemma)


two_dtc_stats = function(dtc1, dtc2) 
#check the similarity of two frequency tables
    { 
    author_stat = function(dtc){
    dtc %>% ungroup() %>% droplevels() %>%
    select(doc_id, term, n) %>%
    complete(doc_id, term, fill = list(n = 0)) %>%
     mutate(no_docs = n_distinct(dtc$doc_id)) %>%
     group_by(doc_id) %>%
     mutate(T.in_doc = sum(n), r = 2*sqrt(n+2/8) / sqrt(T.in_doc)) %>%
     ungroup() %>%
     group_by(term) %>%
     group_by(term, no_docs) %>%
     summarize(mu = mean(r), std = sqrt(var(r)),
           evar = mean(1/T.in_doc), n = sum(n)) %>%
     select(term, no_docs, mu, std, n, evar) %>%
     unique()
    }
    
    anscomb1 <- author_stat(dtc1) %>% filter(n > 3)
    anscomb2 <- author_stat(dtc2) %>% filter(n > 3)

    res <- anscomb1 %>% full_join(anscomb2, by = 'term') %>%
            ungroup() %>%
            fill(no_docs.x, no_docs.y) %>%
            replace(., is.na(.), 0) %>%
            filter(n.x + n.y > 3) %>%
            mutate(var_pooled = ((no_docs.x-1) * std.x^2 + (no_docs.y-1) * std.y^2) / (no_docs.x + no_docs.y-1),
              z_p = (mu.y - mu.x) / sqrt(var_pooled),
              z1 = (mu.y - mu.x) / std.x,
              z2 = (mu.x - mu.y) / std.y,
              tot.x = sum(n.x),
              tot.y = sum(n.y),
              ) %>%
            rowwise() %>%
            mutate(p = ifelse(n.x + n.y > 0, 
                   binom.test(x = n.x, n = n.x+n.y, p = (tot.x-n.x) / (tot.x + tot.y - n.x - n.y))$p.value,
                   1)) %>%
            arrange(abs(p)) %>%
            mutate(rank = order(p)) %>%
            ungroup()
    res
}    

df1 = data_frame()
df0 = data_frame()
#dfA = data_frame()

top_rank = 100
M = 4

nBS = min(990)

lo_classes = unique(anscombe$class)
df_smp <- expand.grid(lo_classes, lo_classes) %>% 
          filter(as.vector(Var2) > as.vector(Var1)) %>%
          sample_n(nBS)

pb <- txtProgressBar(min = 0, max = nBS, style = 3)

for (i in seq(nBS)) {
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

        stat_within <- two_dtc_stats(dtc_doc, dtc_corpus)
        stat_between <- two_dtc_stats(dtc_doc, dtc_other_corpus)

        res0 <- stat_within %>%
                filter(rank < top_rank, n.y > 1) %>%
                arrange(abs(p)) %>%
                mutate(rank = order(p),
                       type = 'within',
                       nstd = std.y / mu.y,
                       class = paste(smp$Var1,smp$Var2)) %>%
                select(rank,p,type,nstd,
                       z = z2, z_p = z_p,
                       std = std.y,
                       mu = mu.y,
                       class,
                       evar = evar.y) 
        
        p_th = HC.vals(res0$p, stbl = TRUE, alpha = 0.25)$p.star
        n_th = sum(res0$p <= p_th)
        res0 <- res0 %>% mutate(n_th = n_th)

        res1 <- stat_between %>%
                filter(rank < top_rank, n.y > 1) %>%
                arrange(abs(p)) %>%
                mutate(rank = order(p),
                       type = 'between',
                       nstd = std.y / mu.y,
                      class = paste(smp$Var1,smp$Var2)) %>%
                select(rank, type, p, nstd,
                       z = z2,
                       z_p = z_p,
                       std = std.y,
                       mu = mu.y,
                      class,
                       evar = evar.y) 

        p_th = HC.vals(res1$p, stbl = TRUE, alpha = 0.25)$p.star
        n_th = sum(res1$p <= p_th)
        
        res1 <- res1 %>% mutate(n_th = n_th)
        
        df1 <- rbind(df1, res1)
        df0 <- rbind(df0, res0)
    }
}
close(pb)


write.csv(df0, paste("./df0_",dataset_name,".csv", sep = ""))
write.csv(df1, paste("./df1_",dataset_name,".csv", sep = ""))

