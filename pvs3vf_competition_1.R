#library
library(readr)  
library(dplyr)  

#Reading and viewing data set
titanic = read_csv("train.csv")
head(titanic)
glimpse(titanic)

#Converting some categorical attributes to factors
titanic$Sex<-as.factor(titanic$Sex)
titanic$Embarked<-as.factor(titanic$Embarked)

#viewing data to ensure that the data type has been changed
glimpse(titanic)

#splitting titanic training data into train and test temporarily where training data is the data with Age populated and
#test contains rows where Age is NULL. Note that we will use linear regression to derive the missing Ages.
test<-titanic[is.na(titanic$Age),]
train<-titanic[!is.na(titanic$Age),]

#Linear Regression Model
lm1 <- lm(Age ~ Pclass+Sex+SibSp, data=train) #Linear Model
anova(lm1,test="Chisq")
summary(lm1)  

#Predicting missing Ages
test$Age <- predict(lm1, newdata=test)
#Recombing the data to recreate the now imputed titanic training data set.
titanic_imputed<-rbind(train,test)
sum(is.na(titanic_imputed$Age))

# Let's subset to cross validate model accuracy and ensure there is no overfititng 
sub <- sample(1:891,size=floor(0.7*891))
train <- titanic_imputed[sub,]     # Select subset for cross-validation
valid <-titanic_imputed[-sub,]

#Performing logistic regression on train using Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
lg <- glm(Survived~Pclass+Sex+Age+SibSp+Parch+Fare+Embarked, data=train, family = "binomial")
summary(lg)

#To improve model performance
anova(lg,test="Chisq")

#Using training (70%) of titanic data to train a logistic model with significant explanatory variables
lg <- glm(Survived~Pclass+Sex+Age+SibSp, data=train, family = "binomial")
summary(lg)# AIC:  Residual Deviance: 

#Checking accuracy on both training data and validation (30%) of the data.
#empty matrix for predictions
preds <- rep(0,623) 
prob = predict(lg, type = 'response')
#assign 1 if probability>0.5

preds[prob>0.5] <- 1 # p>0.5 -> 1
#Confusion Matrix 
cm_train=table(preds,train$Survived)
cm_train
#Accuracy on training data
accuracy=sum(diag(cm_train))/sum(cm_train) #
accuracy

#Testing on validation data
prob_v = predict(lg, type = 'response',newdata = valid)
preds_v <- rep(0,268) 
preds_v[prob_v>0.5] <- 1 # p>0.5 -> 1
#Confusion Matrix 
cm_valid=table(preds_v,actual=valid$Survived)
#Accuracy on validation data
accuracy_v=sum(diag(cm_valid))/sum(cm_valid)
accuracy_v #accuracy 

#We are satisfied with this accuracy and hence train this model using our whole titanic training data
lg <- glm(Survived~Pclass+Sex+Age+SibSp, data=titanic_imputed, family = "binomial")
summary(lg)

#Now we use this model to predict Survived for test data
titanic_test = read_csv("test.csv")
head(titanic_test)
glimpse(titanic_test)
dim(titanic_test)
prob_t = predict(lg, type = 'response',newdata = titanic_test)
titanic_test$Survived<-0
titanic_test$Survived[prob_t>0.5]<-1
write.csv(titanic_test[,c("PassengerId","Survived")], file = "pvs3vf-titanic-predictions.csv",row.names=FALSE)
    