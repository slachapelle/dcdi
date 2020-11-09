# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

source('prepare_pc.R', chdir=T)
source('pc_function.R', chdir=T)
source('indepTests/gaussCIcontexttest.R', chdir=T)
source('indepTests/parcortest.R', chdir=T)
source('indepTests/gaussCIcontexttest_slow.R', chdir=T)
source('indepTests/gaussCIsincontest.R', chdir=T)

pc_wrapper <- function(data,systemVars,contextVars,known,targets=NULL,alpha=1e-2,verbose=0,subsamplefrac=0.0,test='gaussCIcontexttest',obsContext=matrix(0,1,length(contextVars))) {
  # data:          Nxp matrix
  # systemVars:    indices of system variables (in 1..p)
  # contextVars:   indices of context variables (in 1..p)
  # alpha:         p-value threshold for independence test (default: 1e-2)
  # verbose:       verbosity level (default: 0)
  # subsamplefrac: fraction of subsamples used for subsampling; if 0, don't subsample and use all data
  # test:          'gaussCItest' / 'gaussCIcontexttest_slow' / 'gaussCIcontexttest' (only relevant for mode=='jci*')
  #                   gaussCItest: just partial correlations
  #                   gaussCIcontexttest_slow: tests within each context value separately and combines with Fisher's method
  #                   gaussCIcontexttest: tests within each context value separately and combines with Fisher's method, faster implementation
  # obsContext:    vector of values of context variables that defines the observational context

  suppressMessages(library(pcalg))
  suppressMessages(library(mgcv))
  suppressMessages(library(kpcalg))
  mode<-"jci123"

  # prepare data
  X<-prepare_pc(data,systemVars,contextVars,subsamplefrac,'multiple',obsContext=c())
  data<-X$data
  systemVars<-X$systemVars
  contextVars<-X$contextVars
  N<-X$N
  p<-X$p
  removeNAs<-X$removeNAs

  # setup independence test
  if( test == 'kernelCItest' ) {
    indepTest<-kernelCItest
    suffStat<-list(data=data, ic.method="hsic.gamma")
  } else if( test == 'gaussCItest' ) {
    indepTest<-gaussCItest
    suffStat<-list(C=cor(data),n=N,removeNAs=removeNAs)
  } else if( test == 'disCItest' ) {
    indepTest<-disCItest
    suffStat<-list(dm=data,adaptDF=FALSE)
  } else if( test == 'gaussCIcontexttest_slow' ) {
    indepTest<-gaussCIcontexttest_slow
    suffStat<-list(data=data,contextVars=contextVars,verbose=verbose,removeNAs=removeNAs)
  } else if( test == 'gaussCIcontexttest' ) {
    indepTest<-gaussCIcontexttest
    # find indices for unique joint values of context variables
    uniqueContextValues<-uniquecombs(data[,contextVars])
    regimes<-attr(uniqueContextValues,'index')
    suffStat<-list(data=data,contextVars=contextVars,uniqueContextValues=uniqueContextValues,regimes=regimes,verbose=verbose,removeNAs=removeNAs,skiptests=(mode=='jci123r'))
  } else if( test == 'gaussCIsincontest' ) {
    indepTest<-gaussCIsincontest
    suffStat<-list(data=data,contextVars=contextVars,verbose=verbose,removeNAs=removeNAs)
  }

  n_nodes <- dim(data)[2]


  fixedGaps <- NULL
  fixedEdges <- NULL

  cpdag<-pc_modified(labels=colnames(data),suffStat,indepTest,systemVars,contextVars,alpha=alpha,verbose=verbose,fixedGaps=fixedGaps,fixedEdges=fixedEdges, u2pd="relaxed", known=known, targets=targets)
  cpdag <- cpdag@graph

  result<-list(p=p,systemVars=systemVars,contextVars=contextVars,labels=colnames(data),cpdag=cpdag)

  return(result)
}
