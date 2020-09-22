#----
# 1.0 Load packages
#----

library(pdftools)
library(dplyr)
library(ggplot2)
library(wordcloud)
library(tidytext)
library(tm)
library(qdap)
library(stringr)
library(topicmodels)
library(skmeans)
library(cluster)
library(lsa)
library(igraph)
library(reticulate)
library(quanteda)
library(tokenizers)
library(lexRankr)
options(stringsAsFactors = FALSE) 



#----
# 2.0 Data preparation
#----

# 2.1 Load the pdf documents:
# All 20 pdf documents are stored in the 'data' folder
# Extract the names if all the pdf files in the folder:
files.names <- list.files(path ='~/R/Thesis/data', 
                          pattern = "pdf$") 
files.names

# Load the files into a list with 20 elements, each corresponding to 1 transcript:
transcripts <- lapply(paste0('~/R/Thesis/data/', files.names), pdf_text)


# 2.2 Data cleaning: 
# Initialize a data frame for storing cleaned management discussion text and corresponding summaries:
main.df <- data.frame(doc_id = seq(1, length(transcripts), 1), 
                      text = rep(NA, length(transcripts)))

# Initialize a list for storing the Q&A text:
q.a.list <- vector(mode = "list", 
                   length = length(transcripts))
names(q.a.list) <- seq(1, length(transcripts),1)

# A for loop that repeats the cleaning steps over the 20 documents:
for (j in 1:length(transcripts)) {
  
  # Extract the transcript text from the list:
  text <- transcripts[[j]]
  
  # Extract the part of text that lists Company participants:
  company.participants <- ifelse(is.na(str_extract(text, 'Company Participants(.|\r\n)*(?=Other Participants)')),
                                 str_extract(text, 'Company Participants(.|\r\n)*(?=MANAGEMENT DISCUSSION SECTION)'),
                                 str_extract(text, 'Company Participants(.|\r\n)*(?=Other Participants)'))
  # Separate each name in the Company participants list and store as a vector (for later use):
  company.participants <- strsplit(company.participants, 
                                   split = '\r\n.')[[1]][-1] %>%
    gsub('\r\n', '', .) %>%
    trimws(which = "both")
  
  # Remove the header, page numbers, and everything that precedes the utterance of the first speaker:
  regex <- paste0('MANAGEMENT DISCUSSION SECTION(\r\nOperator\r\n)?(.|\r\n)*(?=',company.participants[1],')') 
  text.clean <- text %>% 
    {gsub('Page [0-9][0-9]? of [0-9][0-9]?', ' ', .)} %>% # Remove page numbers
    {gsub('Company Name: .* Current Year: [0-9]*[.][0-9]*', ' ', .)} %>% # Remove header information
    {str_remove(., regex)} %>% # Remove the text corresponding to the utterance of the Operator
    {gsub(str_extract(text, 'Company Participants(.|\r\n)*(?=MANAGEMENT DISCUSSION SECTION)'), ' ', .)} # Remove the text that precedes the management discussion section
  
  # Extract the header information:
  header <- str_extract(text, 'Company Name: (.|\r\n)* Current Year: [0-9]*[.][0-9]*')
  header <- header[1]
  
  # Remove page breaks: 
  text.merged <- paste(text.clean, collapse = '')
  text.merged <- text.merged %>%
    gsub('\r\n ( )* \r\n \r\n', ' ',.)
  
  # Extract Q&A section and remove it from the main text:
  q.a <- str_extract(text.merged, 'Q&A\r\n(.|\r\n)*')
  q.a <- gsub('(\r\nOperator(\r\n)?)|((\r\n)?Operator\r\n)','\n', q.a)
  for (i in 1:length(company.participants)) {
    q.a <- gsub(paste0('\r\n', company.participants[i]), '\n', q.a)
    q.a <- gsub(paste0(company.participants[i],'\r\n'), '\n', q.a)
  }
  q.a <- gsub('\r\n<Q', '\n<Q', q.a)
  q.a <- gsub('\r\n<A', '\n<A', q.a)
  q.a <- gsub('\r\n', ' ', q.a)
  q.a.vector <- strsplit(q.a, split='\n')[[1]]
  q.a.vector <- q.a.vector[grep('<[QA]', q.a.vector)] # Extract only questions and answers
  q.a.list[[j]] <- q.a.vector
  
  # Extract only managements discussion text into a separate string object:
  mgmt.discussion <- gsub('Q&A\r\n(.|\r\n)*', '', text.merged)
  
  
  # Remove headers (e.g. speaker's name, indications that operator is speaking)
  mgmt.discussion <- gsub('MANAGEMENT DISCUSSION SECTION', '\r\n', mgmt.discussion) 
  mgmt.discussion <- gsub('Q[1-9].*Earnings Call\r\n', '\r\n', mgmt.discussion)
  mgmt.discussion <- gsub('\r\nOperator\r\n', '\r\n', mgmt.discussion)
  for (i in 1:length(company.participants)) {
    mgmt.discussion <- gsub(paste0('(\r\n)?', company.participants[i]), '\r\n', mgmt.discussion)
  }
  
  # Remove line breaks in the final text object:
  mgmt.discussion <- gsub('\r\n', ' ', mgmt.discussion)
  
  # Remove redundant white space:
  mgmt.discussion <- stripWhitespace(mgmt.discussion)
  
  # Remove sentences that contain introductory phrases, such as 'Welcome', 'Hello' etc.:
  tokens <- unlist(tokens(mgmt.discussion, what = 'sentence'))
  tokens <- tokens[!grepl("([Tt]hank)|([Ww]elcome)|([Ll]adies(.)*[Gg]entlemen)|([Hh]ello)|([Gg]ood (morning|evening|afternoon))|(Q&A)", tokens)]
  mgmt.discussion <- paste(tokens, collapse=' ')
  
  # Store the final text into the main data frame:
  main.df[j,2] <- mgmt.discussion
}

# 2.3 Save the main data frame into a csv file:
write.csv(main.df, file ='~/R/Thesis/transcripts.csv')

# 2.4 Remove unnecessary objects from the environment:
rm(q.a.list, transcripts, company.participants, files.names, header, i,j, mgmt.discussion,
   q.a, q.a.vector, regex, text, text.clean, text.merged, tokens)



#----
# 3.0 Lexical text summarization (using bag-of-words sentence-vector representation)
#----

# 3.1 Create a column in the main data frame to store the lexical summaries:
main.df$lexical.summary <- rep(NA, nrow(main.df))

# 3.2 A loop which performs lexical summarization on 20 transcripts:
for (i in 1:nrow(main.df)) {
  
  # Create data frame for 1 transcript:
  # Extract the text corresponding to 1 transcript
  df <- data.frame(text = main.df[i,2])
  
  # Tokenize the text on a sentence level
  sentences <- unlist(tokens(df$text[1], what = 'sentence'))
  
  # Remove sentences that contain 3 or less words:
  sentences <- sentences[which(str_count(sentences, '\\w+')>3)]
  
  # Store the original sentences (without pre-processing) for displaying the obtained summaries:
  original.sentences <- sentences
  
  # Replace all numbers with '123':
  for (k in 1:length(sentences)) {
  sentences[k] <- gsub('[0-9]+([,.][0-9]+)?', '123', sentences[k])
  }
  
  # Create a data frame where each row corresponds to a sentence; assign each sentence an ID:
  sentences <- data.frame(sentence_id = seq(1, length(sentences), 1), 
                          sentence = sentences, original.sentence = original.sentences )
  
  # Compute TextRank scores:
  # Note: here, LexRank package was used, which is equivalent to TextRank in its ranking procedure
  # However, Lexrank uses Tf-Idf weighting when obtaining sentence-vector representation, and was therefore used here.
  # The command also applies standard  text pre-processing steps.
  lr <- bind_lexrank(sentences, 
                     sentence, 
                     doc_id = sentence_id, 
                     level = 'sentences',
                     removePunc = TRUE, # Remove punctuation
                     toLower = TRUE, # Set everything to lower case
                     stemWords = TRUE, # Perform stemming
                     rmStopWords = TRUE) %>%  # Remove stop words 
    arrange(desc(lexrank)) # Sort the sentences by their TextRank scores, in descending order
  
  # Extract top 5 sentences into a summary and sort i the original order (by sentence_id):
  summary <- lr %>%
    top_n(5) %>%
    arrange(sentence_id)
  
  # Paste the summary into the main data frame:
  main.df$lexical.summary[i] <- paste(summary[,3], collapse = '\n')
}

# 3.3 The summaries: 
main.df$lexical.summary

# 3.4 Remove unnecessary objects:
rm(df, lr, sentences, summary, i, k, original.sentences)



#----
# 4.0 Semantic summarization algorithm (using Doc2Vec sentence-vector representation)
#----

# 4.1 Obtain Doc2Vec sentence-vector representations
# Create a list: each element will store a data frame corresponding to 1 transcript
df_list <- list()
# Fill the list with data frames
# Each row in a dataframe corresponds to a sentence in the corresponding transcript:
for (i in 1:nrow(main.df)) {
  df <- data.frame(text = main.df[i,2])
  sentences <- unlist(tokens(df$text[1], what = 'sentence')) # Tokenize on a sentence level
  sentences <- data.frame(doc_id = seq(1, length(sentences), 1), 
                          text = sentences) # Create the dataframe with sentences
  sentences <- sentences[which(str_count(sentences$text, '\\w+')>3),] # Remove sentences that contain 3 words or less
  df_list[[i]] <- sentences # Store the data frame into the list
}

# Count the number of sentences in the corpus:
n_sent <- 0
for (i in 1:nrow(main.df)) {
  text <- main.df[i,2]
  sentences <- unlist(tokens(text, what='sentence'))
  sentences <- data.frame(doc_id=seq(1, length(sentences), 1), 
                          text=sentences)
  
  sentences <- sentences[which(str_count(sentences$text, '\\w+')>3),]
  n_sent <- n_sent + length(sentences$text)
}
n_sent # 3568


# Save the data frames corresponding to each transcript as csv files
# Result: 20 files; will be used to obtain sentence embeddings
for (j in 1:length(df_list)) {
  write.csv(file=paste0("~/R/Thesis/transcripts_df/transcript",j, '.csv'), df_list[[j]])
}

# The next steps involve obtaining Doc2Vec sentence-vector representations
# This was done using gensim package, which is only available for Python
# Therefore, the following step will be performed in Python (see Doc2Vec.ipynb file)
# The result is a dataframe with each row representing an embedding of a sentence

# Load the dataframe with embeddings (for the sentences from all transcripts:
embeddings <- read.csv(file = '~/R/Thesis/embeddings.csv')

# Separate the data frame so that embeddings for the sentences for each document appear in a separate data frame:
# Store these data frames in a list
d2v_list <- list()
row <- 1
for (n in 1:nrow(main.df)) {
  # Get the number of rows to extract:
  text <- main.df[n,2] 
  sentences <- unlist(tokens(text, what = 'sentence'))
  sentences <- data.frame(doc_id = seq(1, length(sentences), 1), 
                          text = sentences)
  sentences <- sentences[which(str_count(sentences$text, '\\w+')>3),] 
  nrow <- nrow(sentences)
  
  # Extract the given number of rows from the embeddings data frame:
  #Store then as a data frame in the list element corresponding to the document of interest:
  embeddings_df <- embeddings[row:(row + nrow-1),]
  d2v_list[[n]] <- embeddings_df
  
  # Update the row from which the extraction begins in the next iteration:
  row = row + nrow
}

# 4.2 Generating a semantic summary
# Create a column in the main data frame to store the semantic summaries:
main.df$semantic.summary <- rep(NA, nrow(main.df))

# A loop for generating summaries that iterates over the 20 documents:
for (i in 1:nrow(main.df)) {
  
  # Extract the text of the given transcript: 
  text <- main.df[i,2]
  
  # Tokenize of a sentences level:
  sentences <- unlist(tokens(text, what = 'sentence'))
  
  # Create a dataframe with sentences and their IDs:
  sentences <- data.frame(doc_id = seq(1, length(sentences), 1), 
                          text = sentences)
  # Remove sentences with 3 words or less:
  sentences <- sentences[which(str_count(sentences$text, '\\w+')>3),]
  
  # Extract the embeddings data frame of the corresponding transcript:
  df <- d2v_list[[i]]
  
  # Similarity matrix: contains pairwise similarities between all sentence-embeddings of the transcript:
  df <- t(df) # Transpose the embeddings matrix (since the cosine function calculates pairwise similarities between columns)
  sim.m.sem <- cosine(df)
  
  # In case there are embeddings that contain all 0's, the cosine will be NA; replace all NA's with 0
  sim.m.sem[is.nan(sim.m.sem)] <- 0 
  
  # Creating a graph object:
  graph <- graph_from_adjacency_matrix(sim.m.sem, 
                                       mode = "undirected", 
                                       weighted = TRUE,
                                       add.colnames = NULL,
                                       diag = FALSE)
  
  # PageRank (Same as TextRank): the process of assigning scores to the sentences:
  textrank.scores <- page.rank(graph)
  
  # Append a column to the sentences dataframe with their corresponding TextRank scores:
  sentences$textrank.scores <- textrank.scores$vector
  
  # Extract the sentences for the summary:
  sem.summary <- sentences %>%
    arrange(desc(textrank.scores)) %>% # Sort in descending order (by Textrank scores)
    top_n(5) %>% # Select top 5 sentences
    arrange(doc_id) # Sort in the original order (by sentence ID)
  
  main.df$semantic.summary[i] <- paste(sem.summary$text, collapse = '\n') 
  
} 

# 4.3 The summaries:
main.df$semantic.summary

# 4.4 Remove unnecessary objects:
rm(df, embeddings_df, graph, sem.summary, sentences, sim.m.sem, textrank.scores, 
   i, j, n, n_sent, nrow, row, text)


#----
# 5.0 Qualitative examination of one example transcript:
#----

# 5.1 Extracting the text:
example_id <- 9
example_text <- main.df$text[example_id]

# 5.2 Obtaining sentence-vector representation:
# Bag of words representation:
sentences <- unlist(tokens(example_text, what = 'sentence'))
sentences <- sentences[which(str_count(sentences, '\\w+')>3)] # 152 sentences
sentences <- data.frame(doc_id = seq(1, length(sentences), 1), 
                        text = sentences) # data frame with text corresponding to sentences

# Create document-term matrix with tf-idf weighing where documents=sentences
corpus <- VCorpus(DataframeSource(sentences))
tdm <- TermDocumentMatrix(corpus, control = list(weighting = weightTfIdf,
                                                 removePunctuation = TRUE,
                                                 stopwords = TRUE,
                                                 stemming = TRUE ))
tdm.m <- as.matrix(tdm) # Bag-of-Words representation

# Doc2Vec representation:
d2v.m <- t(as.matrix(d2v_list[[example_id]]))


# 5.3 Obtain similarity matrices:
tdm.m.sim <- cosine(tdm.m)
tdm.m.sim[is.nan(tdm.m.sim)] <- 0 # Bag-of-words
d2v.m.sim <- cosine(d2v.m)
d2v.m.sim[is.nan(d2v.m.sim)] <- 0 # Doc2Vec


# Choose a sentence to explore the most similar sentences:
example_sent_id <- 90  
example_sentence <- sentences[example_sent_id,2]
example_sentence

# Extract the similarities of this sentence with all other sentences from both similarity matrices:
a <- tdm.m.sim[example_sent_id,]
a[example_sent_id] <- 0 # Bag-of-words

b <- d2v.m.sim[example_sent_id,]
b[example_sent_id] <- 0 # Doc2Vec

# Display the sentences:
sentences[example_sent_id,2]
sentences[which.max(a), 2]
sentences[which.max(b), 2]

# 5.4 Obtain example graphs:
graph.a <- graph_from_adjacency_matrix(tdm.m.sim[50:70,50:70], # Choose a subset of 20 sentences
                                       mode = "undirected", # Undirected graph
                                       weighted = TRUE, # Use cosine similarities as weights 
                                       add.colnames = NULL,
                                       diag = FALSE)
graph.b <- graph_from_adjacency_matrix(d2v.m.sim[50:70,50:70], # Choose a subset of 20 sentences 
                                       mode = "undirected", # Undirected graph 
                                       weighted = TRUE,  # Use cosine similarities as weights
                                       add.colnames = NULL,
                                       diag = FALSE)
# Make the edges width 5*weights for visibility:
E(graph.a)$width <- 5*E(graph.a)$weight 
E(graph.b)$width <- 5*E(graph.a)$weight

# Delete edges whose weights are equal to 0:
graph.a <- delete.edges(graph.a, which(E(graph.a)$weight ==0))
graph.b <- delete.edges(graph.b, which(E(graph.b)$weight ==0))

# Plot the graphs:
par(mfrow=c(1,2), mar=c(1,1,1,1), oma=(c(0,0,0,0)))
plot(graph.a, layout=layout_in_circle(graph.a), 
     vertex.size=18, 
     vertex.label.cex=0.8, 
     vertex.label.color='black',
     vertex.frame.color='darkgreen',
     edge.width=E(graph.a)$width,
     vertex.color='white')
title("Summarization \nUsing Bag-of-words", line = -1, cex.main=0.8)

plot(graph.b, layout=layout_in_circle(graph.b), 
     vertex.size=18, 
     vertex.label.cex=0.8, 
     vertex.label.color='black',
     vertex.frame.color='darkgreen',
     vertex.label=seq(50,70,1),
     vertex.color='white',
     edge.width=E(graph.b)$width, 
     edge.color=ifelse(E(graph.b)$weight > 0, "grey","red")) # Make the edges with negative weights grey
title("Summarization \nUsing Doc2Vec", line = -1, cex.main=0.8)


#----
# 6.0 Evaluation results: significance testing:
#----

n <- 120 # Sample size
p <-  71/120 # Proportion of votes given to the semantic algorithm
p_0 <- 0.5 

# Calculate the z-statistic
z_stat <- (p - p_0) / sqrt(p_0 * (1 - p_0) / n)
z_stat # 2.008316

Z <- Normal(0, 1)  # create a standard normal distribution
P_value <- 1 - cdf(Z, z_stat) 
P_value # 0.02




