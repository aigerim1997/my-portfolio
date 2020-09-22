# Packages:
install.packages("poLCA")
library(poLCA)
install.packages("cluster")
library(cluster)

# Load survey Data:
load("SurveyData.RData")

# Descriptive statistics:
summary(SURVEY)


# Function for segment summary:
seg.summ <- function(data, groups) {aggregate(data, list(groups), 
                                              function(x) round(mean(as.numeric(x)), digits=3))
}

# Latent Class Analysis (LCA):
set.seed(02807)

seg.f <- with(SURVEY, 
              cbind(dutch, 
                    education.level, 
                    freq.EUR, 
                    freq.SPAR, 
                    reason, 
                    studyplace, 
                    price.rating, 
                    quality.rating, 
                    walk.rating, 
                    distance.rating, 
                    food.rating)~1 )

# Trying it out with different number of clusters:
seg.LCA2 <- poLCA(seg.f, data=SURVEY, nclass=2 )
seg.LCA3 <- poLCA(seg.f, data=SURVEY, nclass=3 )
seg.LCA4 <- poLCA(seg.f, data=SURVEY, nclass=4 )
seg.LCA5 <- poLCA(seg.f, data=SURVEY, nclass=5 )
seg.LCA6 <- poLCA(seg.f, data=SURVEY, nclass=6 )

# Diagnostics using BIC (determininf the optimal number of clusters:
seg.LCA2$bic # lowest BIC
seg.LCA3$bic
seg.LCA4$bic
seg.LCA5$bic
seg.LCA6$bic


## Summary and plot:

# Number of people in each cluster (total)
table(seg.LCA2$predclass) # 41 and 35 - more balanced
table(seg.LCA3$predclass) # 35, 32 and 9



# Cross-tabs for each variable for describing the clusters:
table(SURVEY$dutch, seg.LCA2$predclass )
table(SURVEY$freq.EUR, seg.LCA2$predclass )
table(SURVEY$freq.SPAR, seg.LCA2$predclass)
table(SURVEY$education.level, seg.LCA2$predclass )
table(SURVEY$studyplace, seg.LCA2$predclass )
table(SURVEY$reason, seg.LCA2$predclass )

# Cross tabs for ratings for describing the clusters:
table(SURVEY$price.rating, seg.LCA2$predclass )
table(SURVEY$quality.rating, seg.LCA2$predclass )
table(SURVEY$distance.rating, seg.LCA2$predclass )
table(SURVEY$walk.rating, seg.LCA2$predclass )
table(SURVEY$food.rating, seg.LCA2$predclass )


# Plot the clusters by projecting on a 2-dimensional plane:
clusplot(SURVEY, 
         seg.LCA2$predclass, 
         color=TRUE, 
         shade=TRUE,
         labels=4, 
         lines=0, 
         main="LCA plot")

