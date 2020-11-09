source('pc_wrapper.R', chdir=T)
library(methods)
library(pcalg)

set.seed(42)

# data: n x d matrix
dataset_raw <- read.csv(file='{FOLDER}{FILE}', sep=",", header=FALSE);
n <- dim(dataset_raw)[1]
d <- dim(dataset_raw)[2]

regimes <- read.csv(file='{FOLDER}{REGIMES}', sep=",", header=FALSE)

# number of context variables
r = max(regimes)
# p = |system var| + |context var|
p = d + r

context <- matrix(0, n, r)
for(i in 1:dim(regimes)[1]){
  regime <- regimes[i,1]
  if (regime != 0){
    context[i, regime] <- 1
  }
}

if({SKELETON}){
  fixedGaps <- read.csv(file='{FOLDER}{GAPS}', sep=",", header=FALSE)
  fixedGaps = (data.matrix(fixedGaps))
  rownames(fixedGaps) <- colnames(fixedGaps)
}else{
  fixedGaps = NULL
}

if({KNOWN}){
  targets <- read.csv(file='{FOLDER}{TARGETS}', sep=",", header=FALSE)
  targets <- (data.matrix(targets))
  show(targets)
}else{
  targets = NULL
}

dataset <- cbind(dataset_raw, context)
  
result <- pc_wrapper(data=dataset, systemVars=1:d, contextVars=(d+1):p, alpha='{ALPHA}', obsContext=matrix(0,1,r), test='{INDEP_TEST}', known='{KNOWN}', targets=targets)
show("pc.R")
show(result$cpdag)
dag <- as(pdag2dag(result$cpdag)[[1]], "matrix")
show(dag)

write.csv(dag, row.names=FALSE, file = '{FOLDER}{OUTPUT}');
