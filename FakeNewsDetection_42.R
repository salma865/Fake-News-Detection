######################################################
###           FAKE NEWS DETECTION PROJECT          ###
######################################################

### install and load needed packages
#####################################
install.packages("dplyr")
install.packages("tidyr")
install.packages("ggplot2")
install.packages("stringr")
install.packages("tm")
install.packages("SnowballC")
install.packages("wordcloud")
install.packages("RColorBrewer")
install.packages("syuzhet")
install.packages("e1071") 
install.packages("randomForest")
library("dplyr")
library("tidyr")
library("ggplot2")
library("stringr")
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library("syuzhet")
library("e1071") 
library("randomForest")


##################
### loading data #
##################


df <- read.csv("E:/R projects/Fake_news_detection/news.csv")

#####################
### 1)Preprocessing #
#####################

#removing coulmn X
df <- df %>% select(-X) 
#drop duplicated rows (29 rows)
df <- df[!duplicated(df),]
#removing null values  (no null values found)
missing_data <- is.na(df) 
df <- na.omit(df)
#encoding label coulmn (0 for fake and 1 for real news)
df$label = factor(df$label,
                  levels = c('FAKE','REAL'),  
                  labels = c(0,1))
df$label

###############################################
### 2)Statistics, analysis and visualizations #
###############################################

# (1)-calculate number of fake and real news,
## visualizing them using pie chart
df %>% group_by(label) %>% summarise(count=n())
counter = c(3152,3154)
label = c("Fake","Real")
pie(counter,labels=label,col=c("red","blue"))

# (2)-calculate number of words in each text in dataset,
## applying some statistic functions(summary-sd-var)
## and visualizing it using box plot
df$words_count <- str_count(df$text, "\\w+")
df %>% group_by(label) %>% summarise(num_of_Words=sum(words_count))
summary(df$words_count)
sd(df$words_count)
var(df$words_count)
cor(df$words_count,as.numeric(df$label),method = "kendall")
boxplot(words_count ~ label,df,ylim=c(0,5000),ylab="words_count",col=c("red","blue"))

# (3)-calculate number of punctiuations in each text in dataset,
## applying some statistic functions(summary-sd-var)
## and visualizing it using box plot
df$punct_count <- lengths(regmatches(df$text, gregexpr("[[:punct:]]+", df$text)))
df %>% group_by(label) %>% summarise(Punct=sum(punct_count))
summary(df$punct_count)
sd(df$punct_count)
var(df$punct_count)
cor(df$punct_count,as.numeric(df$label),method = "kendall")
boxplot(punct_count ~ label,df,ylim=c(0,500),ylab="punct_count",col=c("pink","blue"))

# (4)-get most frequent words in text
## FIRST : cleaning text by creating corpus and applying
## gsup() and tm_map() functions
TextCorpus <- Corpus(VectorSource(df$text))
#function to remove special characters
cleaning <- function(x){ gsub("…|⋆|–|‹|”|“|‘|’|_|/|@|\\|", " ", x) }
TextCorpus <- tm_map(TextCorpus, cleaning)
# Convert the text to lower case
TextCorpus <- tm_map(TextCorpus, content_transformer(tolower))
# Remove numbers
TextCorpus <- tm_map(TextCorpus, removeNumbers)
# Remove english common stopwords
TextCorpus <- tm_map(TextCorpus, removeWords, stopwords("english"))
# Remove punctuations
TextCorpus <- tm_map(TextCorpus, removePunctuation)
# Eliminate extra white spaces
TextCorpus <- tm_map(TextCorpus, stripWhitespace)
# Text stemming - which reduces words to their root form
TextCorpus <- tm_map(TextCorpus, stemDocument)

## SECOND : Build a term document matrix, sort it
## and convert it to  a data frame (each word in text with its frequency)
text_dtm <- TermDocumentMatrix(TextCorpus)
text_dtm_matrix <- as.matrix(text_dtm)
# Sort by descearing value of frequency
text_dtm_sorted <- sort(rowSums(text_dtm_matrix),decreasing=TRUE)
text_dtm_df <- data.frame(word = names(text_dtm_sorted),freq=text_dtm_sorted)
# Display the top 15 most frequent words
head(text_dtm_df, 15)

## THIRD : visualization (bar plot - word cloud)
barplot(text_dtm_df[1:15,]$freq, las = 2, names.arg = text_dtm_df[1:15,]$word,
        col =rainbow(15),ylab = "Word frequencies")

set.seed(1234)
wordcloud(words = text_dtm_df$word, freq = text_dtm_df$freq, min.freq = 6000,
          max.words=150, random.order=FALSE, rot.per=0.5, 
          colors=brewer.pal(8, "Dark2"))

# (5)- applying sentiment analysis on text coulmn
## and visualizing it using bar plot
sentiments <- get_nrc_sentiment(df$text)
sentiments$label <- df$label
summary(sentiments)
barplot(colSums(sentiments[1:10]),las = 2,col = rainbow(10),
        ylab = 'Count',main = 'Sentiment Scores')

#####################################
### 3)Classification and Prediction #
#####################################

### We applyied NaiveBayesin and RandomForest algorithms in 3 different ways ###

## [1] Indebendent variables ---> Title and Text coulmns in news dataset
### FIRST : split data into traning data(75%) and testing data(25%)
set.seed(123)
obs <- nrow(df)
split <- 0.75
training_index <- sample(1:obs, round(obs * split))

y_true <- as.matrix(df$label)
x_training <- df[training_index,1:2]
x_testing <- df[-training_index,1:2]

### SECOND : training and testing NaiveBayes model   (accuracy=50.5%)
NB_model <- naiveBayes(x=x_training,y=as.factor(y_true[training_index]))
NB_model_predict <- predict(NB_model,x_testing)
NB_model_accuracy <- sum(y_true[-training_index] == NB_model_predict)/length(NB_model_predict)*100
NB_model_accuracy

### THIRD : training and testing RandomForest model   (accuracy=52.09%)
RF_model <- randomForest(x=x_training, y=as.factor(y_true[training_index]),ntree = 50)
RF_model_predict <- predict(RF_model, newdata=x_testing)
RF_model_accuracy <- sum(y_true[-training_index] == RF_model_predict)/ length(RF_model_predict)*100
RF_model_accuracy

##########################################################################################################################
## [2] Indebendent variables ---> Sentiment coulmns
### FIRST : split data into traning data(75%) and testing data(25%)
y_true_sentiment <- as.matrix(sentiments$label)
x_training_sentiment <- sentiments[training_index,1:10]
x_testing_sentiment <- sentiments[-training_index,1:10]

### SECOND : training and testing NaiveBayes model   (accuracy=61.29%)
NB_sentiment <- naiveBayes(x=x_training_sentiment,y=as.factor(y_true_sentiment[training_index]))
NB_sentiment_predict <- predict(NB_sentiment,x_testing_sentiment)
NB_sentiment_accuracy <- sum(y_true_sentiment[-training_index] == NB_sentiment_predict)/length(NB_sentiment_predict)*100
NB_sentiment_accuracy

### THIRD : training and testing RandomForest model   (accuracy=66.37%)
RF_sentiment <- randomForest(x=x_training_sentiment, y=as.factor(y_true_sentiment[training_index]),ntree = 50)
RF_sentiment_predict <- predict(RF_sentiment, newdata=x_testing_sentiment)
RF_sentiment_accuracy <- sum(y_true_sentiment[-training_index] == RF_sentiment_predict)/length(RF_sentiment_predict)*100
RF_sentiment_accuracy

##########################################################################################################################
## [3] Indebendent variables ---> text DocumentTermMatrix
### FIRST : split data into traning data(75%) and testing data(25%)

# get the count of words/document
DTM <- DocumentTermMatrix(TextCorpus) 
DTM <-removeSparseTerms(DTM,0.999)
dataset<-as.data.frame(as.matrix(DTM))
dataset$label <-df$label

y_true_dtm <- as.matrix(dataset$label)
x_training_dtm <- dataset[training_index,]
x_testing_dtm <- dataset[-training_index,]
x_training_dtm <- x_training_dtm%>% select(-label)
x_testing_dtm <- x_testing_dtm%>% select(-label)

### SECOND : training and testing NaiveBayes model   (accuracy=72.39%)
NB_dtm <- naiveBayes(x=x_training_dtm , y=as.factor(y_true_dtm[training_index]))
NB_dtm_predict <- predict(NB_dtm,x_testing_dtm)
NB_dtm_accuracy <- sum(y_true_dtm[-training_index] == NB_dtm_predict)/ length(NB_dtm_predict)*100
NB_dtm_accuracy

### THIRD : training and testing RandomForest model   (accuracy=88.57%)
RF_dtm <- randomForest(x=x_training_dtm, y=as.factor(y_true_dtm[training_index]),ntree = 50)
RF_dtm_predict <- predict(RF_dtm,x_testing_dtm)
RF_dtm_accuracy <- sum(y_true_dtm[-training_index] == RF_dtm_predict)/ length(RF_dtm_predict)*100
RF_dtm_accuracy