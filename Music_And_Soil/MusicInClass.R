
library(ggplot2)
train = data.frame(read.csv('music-all.csv'))

train[is.na(train)] = 0

head(train)

comp = prcomp(train[,4:ncol(train)],scale=TRUE, retx=TRUE)
pr = as.data.frame(comp$x)
pr = (pr[,c('PC1','PC2')])

fr = pr
fr$artist = train$artist

plot(comp$sdev)
ggplot(fr,aes(x=PC1, y=PC2)) + geom_point(shape=1)  + geom_text(aes(label=artist), size=5) #


