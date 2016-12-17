##### Preparedata.R
##### PART 1: Prepares raw data to a smaller size for a more feasible manual analysis. Only extracts top 5 answers from each seed.
##### PART 2: BASED ON PREVIOUS ANALYSIS, MERGES EXPLANATIONS SO THE NEXT VERSION OF RAWDATA HAS SOME EXPLANATIONS ALREADY IN IT, REMOVING WORK FOR DUPLICATE ANALYSIS FOR SAME SEED/RESPONSE PAIRS FROM PREVIOUS VERSION.


# *************************************
# PART I. Data Preparation and Export (v2 of rawdata)
# *************************************
wd <- "~/Documents/School/INFO 290/frequency_results"
setwd(wd)

require(dplyr)

#Getting Explanations from understandingMultiplying2 (v2)
um_2 <- read.csv("understandingMultiplying2.csv", stringsAsFactors = F)

first.5.2 <- 
  um_2 %>% 
  group_by(seed) %>% 
  arrange(desc(frequency)) %>% 
  slice(c(1:5))
#This first.5.2 is what we export through this commented command (commented because it will overwrite all the manual analysis)
#write.csv("understandingMultiplying2_top5-analysis.csv")



# *************************************
# PART II. Explanation Merges and v3 Exportation
# *************************************

#Purely reading this in for explanation and issues columns
explanation <- read.csv("understandingMultiplying2_top5-analysis.csv", stringsAsFactors = F)

first.5.2$explanation <- explanation[, "Explanation"]
first.5.2$issues <- explanation[, "Issues"]

# Now creating new first.5 frame for v3 understandingMultiplying
um_3 <- read.csv("understandingMultiplying3.csv", stringsAsFactors = F)
first.5.3 <- 
  um_3 %>% 
  group_by(seed) %>% 
  arrange(desc(frequency)) %>% 
  slice(c(1:5))

#Joining v2 explanation with the same seed/response from v2 to v3 analysis frame.
new.first.5 <- left_join(first.5.3, first.5.2[, c("problem_type", "response", "seed", "explanation", "issues")], by = c("seed", "problem_type", "response"))
# The Commented Command below generates the new analysis file, along with the explanations from the previous version of rawdata.
# write.csv(new.first.5, "understandingMultiplying3_analysis.csv")





# *************************************
# (OPTIONAL) PART III. Exploratory Data Analysis
# *************************************
# hello <- um %>% group_by(seed) %>% summarize(n_distinct(problem_type)) #grouping by problem type
# 
# #Ordered seeds >= 2 problem types, decreasing
# ordered <- hello[hello$`n_distinct(problem_type)` > 1, ][order(hello[hello$`n_distinct(problem_type)` > 1, ]$`n_distinct(problem_type)` , decreasing = T), ]
# 
# #Ordered seeds, ascending
# ordered_asce <- hello[order(hello$`n_distinct(problem_type)` , decreasing = F), ]
# 
# one_seeds <- ordered_asce[ordered_asce$`n_distinct(problem_type)` == 1, ]$seed
# um[um$seed %in% one_seeds, ]
# 
# with_zero <- unique(um[um$problem_type == "0", ]$)
# 
# ordered[ordered$`n_distinct(problem_type)` == 2, ]
# um[um$seed == "xfae7378c5e4cff08", ] %>% group_by(problem_type) %>% arrange(desc(frequency)) %>% slice(c(1:5))
# 
# um[um$seed %in% ordered[ordered$`n_distinct(problem_type)` == 2, ]$seed
