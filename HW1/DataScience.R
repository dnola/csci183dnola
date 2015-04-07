d <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt1.csv"))
head(d)

d$a <- cut(d$Age, c(-Inf,0,18,24,34,44,54,64,Inf))

d$ctr <- d$Clicks/d$Impressions

d$b <- cut(d$ctr, c(-Inf,0,.2,.4,.6,.8,1,Inf))

head(d)

clicks_by_age <- tapply(d$Clicks, d$a, FUN=sum)
barplot(clicks_by_age,names.arg=rownames(clicks_by_age),main="Total Clicks by Age")

ctr_by_age <- tapply(d$ctr, d$a, FUN=mean, na.rm=TRUE)
barplot(ctr_by_age,names.arg=rownames(ctr_by_age), main="Avg CTR by Age")

ctr_by_gender <- tapply(d$ctr, d$Gender, FUN=mean, na.rm=TRUE)
head(ctr_by_gender)
barplot(ctr_by_gender,names.arg=c("female","male"), main = "Avg CTR by Gender")

library("doBy")

stats <- function(x){c(length(x), mean(x),sd(x), min(x), max(x))}
summaryBy(Age~Gender, data=d, FUN=stats)
summaryBy(Age~b, data=d, FUN=stats)
summaryBy(Gender~b, data=d, FUN=mean)

d1 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt1.csv"))
d2 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt2.csv"))
d3 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt3.csv"))
d4 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt4.csv"))
d5 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt5.csv"))
d6 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt6.csv"))
d7 <- read.csv(url("http://stat.columbia.edu/~rachel/datasets/nyt7.csv"))

ds = list(d1,d2,d3,d4,d5,d6,d7)

head(ds[[1]])


for (i in 1:7)
{
  ds[[i]]$a <- cut(ds[[i]]$Age, c(-Inf,0,18,24,34,44,54,64,Inf))
  ds[[i]]$ctr <- ds[[i]]$Clicks/ds[[i]]$Impressions
  ds[[i]]$b <- cut(ds[[i]]$ctr, c(-Inf,0,.2,.4,.6,.8,1,Inf))
}

by_age <- summaryBy(Age~b, data=ds[[1]], FUN=mean)
by_age$day1 <- summaryBy(Age~b, data=ds[[1]], FUN=mean)$Age.mean # I DONT KNOW R WELL ENOUGH TO LOOP THIS
by_age$Age.mean<-NULL
by_age$day2 <- summaryBy(Age~b, data=ds[[2]], FUN=mean)$Age.mean
by_age$day3 <- summaryBy(Age~b, data=ds[[3]], FUN=mean)$Age.mean
by_age$day4 <- summaryBy(Age~b, data=ds[[4]], FUN=mean)$Age.mean
by_age$day5 <- summaryBy(Age~b, data=ds[[5]], FUN=mean)$Age.mean
by_age$day6 <- summaryBy(Age~b, data=ds[[6]], FUN=mean)$Age.mean
by_age$day7 <- summaryBy(Age~b, data=ds[[7]], FUN=mean)$Age.mean

print(by_age)
plot( c(1,2,3,4,5,6,7), data.matrix(by_age[6,2:8]), main="Average age of heavy clickers by day")


by_gender <- summaryBy(Gender~b, data=ds[[1]], FUN=mean)
by_gender$day1 <- summaryBy(Gender~b, data=ds[[1]], FUN=mean)$Gender.mean # I DONT KNOW R WELL ENOUGH TO LOOP THIS
by_gender$Gender.mean<-NULL
by_gender$day2 <- summaryBy(Gender~b, data=ds[[2]], FUN=mean)$Gender.mean
by_gender$day3 <- summaryBy(Gender~b, data=ds[[3]], FUN=mean)$Gender.mean
by_gender$day4 <- summaryBy(Gender~b, data=ds[[4]], FUN=mean)$Gender.mean
by_gender$day5 <- summaryBy(Gender~b, data=ds[[5]], FUN=mean)$Gender.mean
by_gender$day6 <- summaryBy(Gender~b, data=ds[[6]], FUN=mean)$Gender.mean
by_gender$day7 <- summaryBy(Gender~b, data=ds[[7]], FUN=mean)$Gender.mean

print(by_gender)
plot( c(1,2,3,4,5,6,7), data.matrix(by_gender[6,2:8]), main="Female/Male ratio of heavy clickers by day")
      
