#install.packages('vcd')
library(vcd)
# I get slightly different results because I screen out people who didn't answer BOTH questions with a definitive categorical fit

gss = data.frame(read.csv('GSS.csv'))
head(gss)
gss_rel = gss[gss$Gss.year.for.this.respondent==2010 | gss$Gss.year.for.this.respondent==2012,]

tail(gss_rel)

gss_rel['pro_income_equality'] = -1
gss_rel['pro_right_to_gay_marriage'] = -1

gss_rel[gss_rel$Should.govt.reduce.income.differences=='Govt reduce diff' | gss_rel$Should.govt.reduce.income.differences=='2' | gss_rel$Should.govt.reduce.income.differences=='3',]$pro_income_equality = 1
gss_rel[gss_rel$Should.govt.reduce.income.differences=='No govt action' | gss_rel$Should.govt.reduce.income.differences=='6' | gss_rel$Should.govt.reduce.income.differences=='5',]$pro_income_equality = 0

gss_rel[gss_rel$Homosexuals.should.have.right.to.marry=='Strongly agree' | gss_rel$Homosexuals.should.have.right.to.marry=='Agree',]$pro_right_to_gay_marriage = 1
gss_rel[gss_rel$Homosexuals.should.have.right.to.marry=='Strongly disagree' | gss_rel$Homosexuals.should.have.right.to.marry=='Disagree' ,]$pro_right_to_gay_marriage = 0

gss_rel['valid'] = 1

gss_rel[gss_rel$pro_income_equality==-1 | gss_rel$pro_right_to_gay_marriage==-1,]$valid = 0

gss_val = gss_rel[gss_rel$valid==1,]
gss_f = subset(gss_val, select=c("pro_income_equality", "pro_right_to_gay_marriage"))
head(gss_f,30)

tbl = table(gss_f)

labs = round(prop.table(tbl), 3)*100
labs[1] = paste(labs[1], "% Conservative")
labs[2] = paste(labs[2], "% Hardhats")
labs[3] = paste(labs[3], "% Libertarian")
labs[4] = paste(labs[4], "% Liberal")

mosaic(tbl, pop = FALSE, legend=FALSE, gp = gpar(fill=matrix(c("brown1","gold", "light green", "deepskyblue"), 2, 2)))
labeling_cells(text = labs, margin = 0)(tbl)

