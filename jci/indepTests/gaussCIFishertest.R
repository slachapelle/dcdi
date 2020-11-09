# Copyright (c) 2018-2020, Joris M. Mooij. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

gaussCIFishertest <- function(x, y, S, suffStat) {
  stopifnot(length(x) == 1 && length(y) == 1)
  stopifnot(length(intersect(x,y)) == 0)
  stopifnot(length(intersect(union(x,y),S)) == 0)
  verbose<-suffStat$verbose
  N<-length(suffStat$ns)
  stopifnot(N >= 1)
  pvalvec<-vector(mode='double',length=N)
  for( regime in 1:N ) {
    pvalvec[regime]<-gaussCItest(x,y,S,suffStat=list(C=suffStat$Cs[[regime]],n=suffStat$ns[[regime]]))
  }
  stat<-(-2.0) * sum(log(pvalvec))
  # distributed as chi^2 with 2*N d.o.f.
  pval<-pchisq(stat, df=2*N, lower.tail=FALSE)
  if( verbose ) {
    cat(pvalvec,': ',stat,'->',pval,'\n')
  }
  pval
}
