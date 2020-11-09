# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

fisher <- function(data,systemVars,contextVars,alpha=1e-2,verbose=0,subsamplefrac=0.0) {
  # data:          Nxp matrix
  # systemVars:    indices of sysem variables (in 1..p)
  # contextVars:   indices of context variables (in 1..p)
  # alpha:         p-value threshold for independence test (default: 1e-2)
  # verbose:       verbosity level (default: 0)
  # subsamplefrac: fraction of subsamples used for subsampling; if 0, don't subsample and use all data

  suppressMessages(library(pcalg))
  suppressMessages(library(mgcv))

  # prepare data
  X<-prepare_jci(data,systemVars,contextVars,subsamplefrac,'multiple',obsContext=c())
  data<-X$data
  systemVars<-X$systemVars
  contextVars<-X$contextVars
  N<-X$N
  p<-X$p
  removeNAs<-X$removeNAs
  arel<-matrix(0,p,p)
  colnames(arel)<-colnames(data)

  if( length(contextVars > 0) ) {
    # run Fisher's method
    pSys<-length(systemVars)
    pCon<-length(contextVars)
    pvals<-matrix(0,pCon,pSys)

    # find indices for unique joint values of context variables
    uniqueContextValues<-uniquecombs(data[,contextVars])
    regimes<-attr(uniqueContextValues,'index')
    suffStat<-list(data=data,contextVars=contextVars,uniqueContextValues=uniqueContextValues,regimes=regimes,verbose=verbose,removeNAs=removeNAs)
    for( c in 1:pCon ) {
      for( i in 1:pSys ) {
        pvals[c,i]<-gaussCIcontexttest(contextVars[c],i,setdiff(contextVars,contextVars[c]),suffStat=suffStat)
        arel[contextVars[c],i] <- -log(pvals[c,i]) + log(alpha)
      }
    }
  }

  # return result
  result<-list(p=p,systemVars=systemVars,contextVars=contextVars,labels=colnames(data),arel=arel)
}
