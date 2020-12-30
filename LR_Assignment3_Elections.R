library(data.table)
library(ggplot2)

Elections <- read.csv("C:/RAVI/Data science/Assignments/Module 9 Logistic regression/LR Assignment dataset3/election_data.csv/election_data.csv")
View(Elections)

sum(is.na(Elections))
Elections <- na.omit(Elections)

colnames(Elections) <- c("Election_id","Result","Year","Amount_spend","Popularity_Rank")
View(Elections)
attach(Elections)
summary(Elections)

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
#(GLM)generalised linear model

model <- glm(Result ~  Election_id + Year + Amount_spend + Popularity_Rank, data = Elections, family = "binomial")
summary(model)
# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))


model2 <- glm(formula = Result ~ Year + Popularity_Rank, family = "binomial", 
    data = Elections)
summary(model2)

library(MASS)
stepAIC(model)

# Confusion matrix table 
prob <- predict(model,Elections,type="response")
prob
# We are going to use NULL and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Elections$Result)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 


# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob > 0.5, 1, 0)
yes_no <- ifelse(prob > 0.5,"yes","no")

# Creating new column to store the above values
Elections[ , "prob"] <- prob
Elections[ , "pred_values"] <- pred_values
Elections[ , "yes_no"] <- yes_no

View(Elections[ , c(2, 6:7)])

table(Elections$Result, Elections$pred_values)
# Calculate the below metrics
# precision | recall | True Positive Rate | False Positive Rate | Specificity | Sensitivity


# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic

install.packages("ROCR")
library(ROCR)
rocrpred <- prediction(prob, Elections$Result)
rocrperf<-performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained

str(rocrperf)
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)

rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)

# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)
