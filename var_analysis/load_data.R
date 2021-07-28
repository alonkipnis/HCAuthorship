#auxiliary functions
library(readr)
library(tidytext)

read_data_dtc = function(path_to_data) {
# returns a data frame with columns:
# doc_id, author, lemma, n

#PAN in doc_term_count format
dtc_raw = read.csv(path_to_data)

dataset_name = "PAN2019_DTC"

# remove proper names, punctuation, and cardinal numbers
dtc_red <- dtc_raw %>%
           filter(author != '<UNK>') %>%
           filter(POS != 'PROPN') %>%
           filter(POS != 'PUNCT') %>%
           filter(POS != 'NUM') %>%
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

dtc_red
}