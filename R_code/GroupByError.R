##### Common Errors.R
##### Extracts (seed,response) and groups them based on their misconception reason.
##### Result: Can use this (seed, response) grouped by misconceptions to better our hyperparameters

# *************************************
# I. Reading and Morphing Data
# *************************************
library(dplyr)

wd <- "~/Documents/School/INFO 290/frequency_results"
setwd(wd)

#Importing Original Raw Data to get the responses due to Excel Formatting Issues
expla.data <- read.csv("understandingMultiplying3.csv", stringsAsFactors = F)
expla.five <- 
  expla.data %>% 
  group_by(seed) %>% 
  arrange(desc(frequency)) %>% 
  slice(c(1:5))


#Reading in our added explanation/analysis dataset (which was done on the expla.five data)
data <- read.csv("understandingMultiplying3_analysis.csv", stringsAsFactors = F)

#Checking whether the idea of just replacing explanations make sense (The data have to be the same number of rows)
dim(expla.five)
dim(data)



# *************************************
# II. Cleaning Data
# *************************************
#Remove unnecessary Columns
data <- data[, -1]

#Replace Response
data$response <- expla.five$response #because of Excel Formatting Issues

#Explanations Things we don't really care about when maximizing our hyperparameters
toMatch <- c("correct", "Correct", "no idea", "typo", "Typo", "Entering space to get hint", "entering space to get hint", " ", "", "Not follow directions, but correct answer", "not follow directions")
data <- data[!data$explanation %in% toMatch, ]

#Cleaning up explanations a little
data$explanation <- gsub("Typo \\+", "", data$explanation)
data$explanation <- trimws(data$explanation)




# *************************************
# III. Grouping Data
# *************************************
#Creating Groups
groups <- unique(data$explanation) #too many, there are some explanations that are similar in terms of the misconceptions; let's make the size smaller.

# Different Major Groups
reduce.error <- groups[grep("incorrect reduction", groups, ignore.case = T)]
reduce.error <- c(inco.reduce, 
                 groups[grep('simplifying', groups, ignore.case = T)])
d.mult.error <- grep("multiply denominator", groups, ignore.case = T, value = T)
n.mult.error <- grep("multiply numerator", groups, ignore.case = T, value = T)
add.error <- grep("add", groups, ignore.case = T, value = T)
squares.error <- grep("squares", groups, ignore.case = T, value = T)
visual.error <- grep("Incorrect visualization of fractional parts", groups, ignore.case = T, value = T)
frac.of.frac.error <- grep("Misunderstanding of taking fraction", groups, ignore.case = T, value = T)

major.groups <- list(reduce.error, d.mult.error, n.mult.error, add.error, squares.error, visual.error, frac.of.frac.error)
names(major.groups) <- c("reduce", "d.mult", "n.mult", "add", "squares", "visual", "frac.of.frac")


#Adding the major expla.group based on the common misconceptions to data
reduce.index <- which(data$explanation %in% reduce.error)
data[reduce.index, "expla.group"] <- "reduce"
data[which(data$explanation %in% d.mult.error), "expla.group"] <- "d.mult"
data[which(data$explanation %in% n.mult.error), "expla.group"] <- "n.mult"
data[which(data$explanation %in% add.error), "expla.group"] <- "add"
data[which(data$explanation %in% squares.error), "expla.group"] <- "squares"
data[which(data$explanation %in% visual.error), "expla.group"] <- "visual"
data[which(data$explanation %in% frac.of.frac.error), "expla.group"] <- "frac.of.frac"
data[which(is.na(data$expla.group)), "expla.group"] <- data[which(is.na(data$expla.group)), "explanation"] 


#Now exporting seed,response by groups (of problem types)
dir.create("../seedranks")
types <- unique(data$problem_type)
for(t in types){
  rel.data <- data[data$problem_type == t, ]
  sort.rel.data <- rel.data[order(rel.data$expla.group), ]
  write.csv(sort.rel.data, paste0("../seedranks/", t, "_seedresps.csv"))
}

# PREVIOUS CODE: GROUPING BY ERROR
# for(n in names(major.groups)){
#   explas <- unlist(major.groups[n])
#   group.data <- data[data$explanation %in% explas, c("problem_type","seed", "response", "explanation")]
#   group.data <- group.data[order(group.data$problem_type), ]
#   write.csv(group.data, paste0("../seedranks/", n, "_error_seedresps.csv"))
# }
