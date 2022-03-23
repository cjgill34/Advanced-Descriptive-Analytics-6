library(dplyr)
library(caret)
library(dummies)
library(class)
library(FNN)

#Question 7.2 part a
bank = read.csv("/Users/Cassie Gill/OneDrive/SCMA 854/CGill HW6/SCMA 854 HW6/UniversalBank.csv")
bank$Education = as.factor(bank$Education)

bank.dummy = dummy.data.frame(select(bank,-c(ZIP.Code,ID)))
bank.dummy$Personal.Loan = as.factor(bank.dummy$Personal.Loan)
bank.dummy$CCAvg = as.integer(bank.dummy$CCAvg)

set.seed(1)
train.index <- sample(row.names(bank.dummy), 0.6*dim(bank.dummy)[1])
test.index <- setdiff(row.names(bank.dummy), train.index) 
train.df <- bank.dummy[train.index, ]
valid.df <- bank.dummy[test.index, ]

new.df = data.frame(Age = as.integer(40), Experience = as.integer(10), Income = as.integer(84), Family = as.integer(2), CCAvg = as.integer(2), Education1 = as.integer(0), Education2 = as.integer(1), Education3 = as.integer(0), Mortgage = as.integer(0), Securities.Account = as.integer(0), CD.Account = as.integer(0), Online = as.integer(1), CreditCard = as.integer(1))


norm.values <- preProcess(train.df[, -c(10)], method=c("center", "scale"))
train.df[, -c(10)] <- predict(norm.values, train.df[, -c(10)])
valid.df[, -c(10)] <- predict(norm.values, valid.df[, -c(10)])
new.df <- predict(norm.values, new.df)

knn.1 <- knn(train = train.df[,-c(10)],test = new.df, cl = train.df[,10], k=5, prob=TRUE)
knn.attributes <- attributes(knn.1)
knn.attributes[1]

knn.attributes[3]

#The customer would be classified as a 0.


#Question 7.2 part b
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

for(i in 1:14) {
  knn.2 <- knn(train = train.df[,-10],test = valid.df[,-10], cl = train.df[,10], k=i, prob=TRUE)
  accuracy.df[i, 2] <- confusionMatrix(knn.2, valid.df[,10])$overall[1]
}
accuracy.df

# The best choice of k that balances between overfitting 
#and ignoring the predictor information is k=3.

#Question 7.2 part c
knn.3 <- knn(train = train.df[,-10],test = valid.df[,-10], cl = train.df[,10], k=3, prob=TRUE)
confusionMatrix(knn.3, valid.df[,10])

#Question 7.2 part d
customer.df= data.frame(Age = 40, Experience = 10, Income = 84, Family = 2, CCAvg = 2, Education_1 = 0, Education_2 = 1, Education_3 = 0, Mortgage = 0, Securities.Account = 0, CD.Account = 0, Online = 1, CreditCard = 1)
knn.4 <- knn(train = train.df[,-10],test = customer.df, cl = train.df[,10], k=3, prob=TRUE)
knn.4

#The customer would be classified as a 1 based on k=3.

#Question 7.2 part e
bank.dummy = dummy.data.frame(select(bank,-c(ZIP.Code,ID)))
bank.dummy$Personal.Loan = as.factor(bank.dummy$Personal.Loan)
bank.dummy$CCAvg = as.integer(bank.dummy$CCAvg)

set.seed(1)
train.index <- sample(rownames(bank.dummy), 0.5*dim(bank.dummy)[1])  ## need to look at hints
set.seed(1)
valid.index <- sample(setdiff(rownames(bank.dummy),train.index), 0.3*dim(bank.dummy)[1])
test.index = setdiff(rownames(bank.dummy), union(train.index, valid.index))

train.df <- bank.dummy[train.index, ]
valid.df <- bank.dummy[valid.index, ]
test.df <- bank.dummy[test.index, ]

norm.values <- preProcess(train.df[, -c(10)], method=c("center", "scale"))
train.df[, -c(10)] <- predict(norm.values, train.df[, -c(10)])
valid.df[, -c(10)] <- predict(norm.values, valid.df[, -c(10)])
test.df[,-c(10)] <- predict(norm.values, test.df[,-c(10)])

testknn <- class::knn(train = train.df[,-c(10)],test = test.df[,-c(10)], cl = train.df[,10], k=3, prob=TRUE)
validknn <- class::knn(train = train.df[,-c(10)],test = valid.df[,-c(10)], cl = train.df[,10], k=3, prob=TRUE)
trainknn <- class::knn(train = train.df[,-c(10)],test = train.df[,-c(10)], cl = train.df[,10], k=3, prob=TRUE)

confusionMatrix(testknn, test.df[,10])
confusionMatrix(validknn, valid.df[,10])
confusionMatrix(trainknn, train.df[,10])

#The train data has the highest accuracy at .978 followed
#by the valid at .9667 and the test at .964. The train data
#had the largest amount of data to learn from and so it allowed
#for the best results. 

