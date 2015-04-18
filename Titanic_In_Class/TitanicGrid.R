library(rpart)
#install.packages('doMC')
library(caret)

library(doMC)
registerDoMC(cores = 8)


# Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

train = read.csv('train.csv')
class(train$Ticket)

head(train)
age_imp = mean(as.double(train$Age), na.rm=T)
fare_imp = mean(as.double(train$Fare), na.rm=T)
train[is.na(train$Age),]$Age = age_imp
#train[is.na(train$Fare),]$Fare = fare_imp


test = read.csv('test.csv')
head(test)
age_imp = mean(as.double(test$Age), na.rm=T)
fare_imp = mean(as.double(test$Fare), na.rm=T)
test[is.na(test$Age),]$Age = age_imp
test[is.na(test$Fare),]$Fare = fare_imp

fit = rpart(Survived ~ Pclass + Age + Sex + SibSp + Parch + Fare + Embarked, method="class", data=train)
plotcp(fit)
plot(fit, uniform=TRUE)
text(fit, use.n=TRUE, all=TRUE, cex=.8)

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)

gbmGrid <-  expand.grid(interaction.depth = c(1,2, 3, 4), n.trees = seq(200, 8000, 200), shrinkage = c(.05,.01,.005) )

fit <- train(Survived ~ Pclass + Age + Sex + SibSp + Parch + Fare + Embarked, data = train, method = "gbm", trControl = fitControl, tuneGrid =  gbmGrid, verbose=FALSE)

print(fit)
fit
trellis.par.set(caretTheme())
plot(fit) 
summary(fit)

test$Survived = round(predict(fit,newdata=test), digits=0)
towrite = test[c("PassengerId","Survived")]

write.csv(towrite, file="submit.csv", row.names = FALSE)

