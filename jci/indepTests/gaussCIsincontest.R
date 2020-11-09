# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

gaussCIsincontest <- function(x, y, S, suffStat) {
  # suffStat<-list(data=data,contextVars=contextVars,verbose=verbose,removeNAs=removeNAs)

  data<-suffStat$data
  p<-dim(data)[2]
  contextVars<-suffStat$contextVars
  verbose<-suffStat$verbose
  removeNAs<-suffStat$removeNAs

  if( verbose ) 
    cat( 'Testing ', x, '_||_', y, '|', S, '\n' )
  stopifnot(length(x) == 1 && length(y) == 1)
  stopifnot(length(intersect(x,y)) == 0)
  stopifnot(length(intersect(union(x,y),S)) == 0)
  stopifnot(length(intersect(union(c(x,y),S),contextVars)) <= 1)
  tmp<-intersect(union(c(x,y),S),contextVars)
  if( length(tmp) == 0 )
    regimeVar<-0
  else
    regimeVar<-tmp[1]

  if( regimeVar == 0 ) {
    if( verbose )
      cat('regimeVar not in {x,y} U S\n')
    pval <- parcortest(x,y,S,data,removeNAs=removeNAs,verbose=FALSE)
  } else {
    uniqueRegimeValues<-uniquecombs(data[,regimeVar])
    regimes<-attr(uniqueRegimeValues,'index')
    nRegimes<-max(regimes)

    if( x == regimeVar || y == regimeVar ) {
      if( verbose )
        cat('regimeVar in {x,y}\n')
      if( y == regimeVar ) { # swap x, y
        tmp<-x
        x<-y
        y<-tmp
      }

      dataf<-data.frame(data[,union(y,S)])
      cn<-colnames(dataf)
      for(k in 1:length(cn)) {
        cn[k]<-paste('X',k,sep='')
      }
      colnames(dataf)<-cn
      if( verbose ) {
        cat(paste('X',which(union(y,S)==y)[1],' ~ .',sep=''),'\n')
      }
      linm <- glm(as.formula(paste('X',which(union(y,S)==y)[1],' ~ .',sep='')),data=dataf,control=glm.control(maxit=10,epsilon=10^(-6)))
      resid <- residuals(linm)

      pvalvec <- numeric(nRegimes)
      for (R in 1:nRegimes) {
#        inds_R <- which(data[,regimeVar] == uniqueRegimeValues[R])
        inds_R <- which(regimes == R)
        x1 <- resid[inds_R]
        x2 <- resid[-inds_R]
        pvalvec[R] <- 2*min(t.test(x1,x2)$p.value, var.test(x1,x2)$p.value)
      }
      if( verbose ) {
        cat('pvalvec: ',pvalvec,'\n')
      }
      pval <- min(pvalvec)*(nRegimes-1)
      pval <- min(1,pval)
    } else if( regimeVar %in% S ) {
      if( verbose )
        cat('regimeVar in S\n')
      pvalvec <- numeric(nRegimes)
      for (R in 1:nRegimes) {
#        inds_R <- which(data[,regimeVar] == R)
        inds_R <- which(regimes == R)
        pvalvec[R] <- parcortest(x,y,setdiff(S,regimeVar),data[inds_R,],removeNAs=removeNAs,verbose=FALSE)
      }
      stat<-(-2.0) * sum(log(pvalvec))
      # distributed as chi^2 with 2*nRegimes d.o.f.
      pval<-pchisq(stat, df=2*nRegimes, lower.tail=FALSE)
      if( verbose ) {
        cat(pvalvec,': ',stat,'->',pval,'\n')
      }
    }
  }
  if( verbose ) 
    cat( 'p-value:', pval, '\n' )
  
  pval
}
