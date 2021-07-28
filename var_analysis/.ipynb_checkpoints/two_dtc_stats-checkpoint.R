library(tidyverse)

two_dtc_stats = function(dtc1, dtc2, anscomb_param, min_cnt) 
#check the similarity of two frequency tables
    { 
    author_stat = function(dtc){
    dtc %>% ungroup() %>% droplevels() %>%
    select(doc_id, term, n) %>%
    complete(doc_id, term, fill = list(n = 0)) %>%
     mutate(no_docs = n_distinct(dtc$doc_id)) %>%
     group_by(doc_id) %>%
     mutate(T.in_doc = sum(n), r = 2*sqrt(n + anscomb_param) / sqrt(T.in_doc)) %>%
     ungroup() %>%
     group_by(term) %>%
     group_by(term, no_docs) %>%
     summarize(mu = mean(r), std = sqrt(var(r)),
           evar = mean(1/T.in_doc), n = sum(n)) %>%
     select(term, no_docs, mu, std, n, evar) %>%
     unique()
    }
    
    anscomb1 <- author_stat(dtc1) %>% filter(n > min_cnt)
    anscomb2 <- author_stat(dtc2) %>% filter(n > min_cnt)

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
