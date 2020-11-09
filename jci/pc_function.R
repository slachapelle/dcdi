library(pcalg)
library(graph)

pc_modified <- function(suffStat, indepTest, systemVars, contextVars, alpha, labels, p, known, targets,
               fixedGaps = NULL, fixedEdges = NULL, NAdelete = TRUE, m.max = Inf,
               u2pd = c("relaxed", "rand", "retry"),
               skel.method = c("stable", "original", "stable.fast"),
               conservative = FALSE, maj.rule = FALSE,
               solve.confl = FALSE, numCores = 1, verbose = FALSE)
{
  ## Purpose: Perform PC-Algorithm, i.e., estimate skeleton of DAG given data
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## - dm: Data matrix (rows: samples, cols: nodes)
  ## - C: correlation matrix (only for continuous)
  ## - n: sample size
  ## - alpha: Significance level of individual partial correlation tests
  ## - corMethod: "standard" or "Qn" for standard or robust correlation
  ##              estimation
  ## - G: the adjacency matrix of the graph from which the algorithm
  ##      should start (logical)
  ## - datatype: distinguish between discrete and continuous data
  ## - NAdelete: delete edge if pval=NA (for discrete data)
  ## - m.max: maximal size of conditioning set
  ## - u2pd: Function for converting udag to pdag
  ##   "rand": udag2pdag
  ##   "relaxed": udag2pdagRelaxed
  ##   "retry": udag2pdagSpecial
  ## - gTrue: Graph suffStatect of true DAG
  ## - conservative: If TRUE, conservative PC is done
  ## - numCores: handed to skeleton(), used for parallelization
  ## ----------------------------------------------------------------------
  ## Author: Markus Kalisch, Date: 26 Jan 2006; Martin Maechler
  ## Modifications: Sarah Gerster, Diego Colombo, Markus Kalisch
  show("========== TARGETS ============")
  show(targets)

  ## Initial Checks
  cl <- match.call()
  if(!missing(p)) stopifnot(is.numeric(p), length(p <- as.integer(p)) == 1, p >= 2)
  if(missing(labels)) {
    if(missing(p)) stop("need to specify 'labels' or 'p'")
    labels <- as.character(seq_len(p))
  } else { ## use labels ==> p  from it
    stopifnot(is.character(labels))
    if(missing(p)) {
      p <- length(labels)
    } else if(p != length(labels))
      stop("'p' is not needed when 'labels' is specified, and must match length(labels)")
    else
      message("No need to specify 'p', when 'labels' is given")
  }

  # added
  fixedEdges <- matrix(FALSE, p, p)
  fixedGaps <- matrix(FALSE, p, p)
  if(length(contextVars) > 0){
      # fix edges between contextVars
     fixedEdges[contextVars,contextVars] <- TRUE
     fixedGaps[contextVars,contextVars] <- TRUE
     if(known){
        # fix edges between contextVars and systemVars
        fixedEdges[systemVars,contextVars] <- TRUE
        fixedEdges[contextVars,systemVars] <- TRUE
        fixedGaps[systemVars,contextVars] <- TRUE
        fixedGaps[contextVars,systemVars] <- TRUE
     }
  }


  u2pd <- match.arg(u2pd)
  skel.method <- match.arg(skel.method)
  if(u2pd != "relaxed") {
    if (conservative || maj.rule)
      stop("Conservative PC and majority rule PC can only be run with 'u2pd = relaxed'")

    if (solve.confl)
      stop("Versions of PC using lists for the orientation rules (and possibly bi-directed edges)\n can only be run with 'u2pd = relaxed'")
  }

  if (conservative && maj.rule) stop("Choose either conservative PC or majority rule PC!")

  ## Skeleton
  # m.max = 1
  skel <- skeleton(suffStat, indepTest, alpha, labels = labels, method = skel.method,
                   fixedGaps = fixedGaps, fixedEdges = fixedEdges,
                   NAdelete=NAdelete, numCores=numCores, verbose=TRUE)
  skel@call <- cl # so that makes it into result

  # added: Orient edges between context and system variables
  g <- as(skel@graph, "matrix")
  g[systemVars,contextVars] <- 0

  g[contextVars, contextVars] <- 1
  for (i in contextVars){
      g[i,i] <- 0
  }
  if(known){
      # g[contextVars, systemVars] = targets
      g[contextVars, systemVars] <- 0
      j <- 1
      for (i in contextVars){
          g[i, systemVars[j]] <- 1
          j <- j + 1
      }
  }
  skel@graph <- as(g, "graphNEL")


  ## Orient other edges
  if (!conservative && !maj.rule) {
    switch (u2pd,
            "rand" = udag2pdag(skel),
            "retry" = udag2pdagSpecial(skel)$pcObj,
            "relaxed" = udag2pdagRelaxed(skel, contextVars, verbose = verbose, solve.confl = solve.confl))
  }
  else { ## u2pd "relaxed" : conservative _or_ maj.rule

    ## version.unf defined per default
    ## Tetrad CPC works with version.unf=c(2,1)
    ## see comment on pc.cons.intern for description of version.unf
    pc. <- pc.cons.intern(skel, suffStat, indepTest, alpha,
                          version.unf = c(2,1), maj.rule = maj.rule, verbose = verbose)
    udag2pdagRelaxed(pc.$sk, verbose = verbose,
                     unfVect = pc.$unfTripl, solve.confl = solve.confl)
  }
} ## {pc}


##################################################
## udag2pdag
##################################################
udag2pdag <- function(gInput, verbose = FALSE) {
  ## Purpose: Transform the Skeleton of a pcAlgo-object to a PDAG using
  ## the rules of Pearl. The output is again a pcAlgo-object.
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## - gInput: pcAlgo object
  ## - verbose: 0 - no output, 1 - detailed output
  ## ----------------------------------------------------------------------
  ## Author: Markus Kalisch, Date: Sep 2006, 15:03

  res <- gInput
  if (numEdges(gInput@graph) > 0) {
    g <- as(gInput@graph,"matrix") ## g_ij if i->j
    p <- as.numeric(dim(g)[1])
    pdag <- g
    ind <- which(g == 1,arr.ind = TRUE)

    ## Create minimal pattern
    for (i in seq_len(nrow(ind))) {
      x <- ind[i,1]
      y <- ind[i,2]
      allZ <- setdiff(which(g[y,] == 1),x) ## x-y-z
      for (z in allZ) {
        if (g[x,z] == 0  &&
            !(y %in% gInput@sepset[[x]][[z]] ||
                y %in% gInput@sepset[[z]][[x]])) {
          if (verbose) {
            cat("\n",x,"->",y,"<-",z,"\n")
            cat("Sxz=",gInput@sepset[[z]][[x]],"Szx=",gInput@sepset[[x]][[z]])
          }
          pdag[x,y] <- pdag[z,y] <- 1
          pdag[y,x] <- pdag[y,z] <- 0
        }
      }
    }

    ## Test whether this pdag allows a consistent extension
    res2 <- pdag2dag(as(pdag,"graphNEL"))

    if (res2$success) {
      ## Convert to complete pattern: use rules by Pearl
      old_pdag <- matrix(0, p,p)
      while (!all(old_pdag == pdag)) {
        old_pdag <- pdag
        ## rule 1
        ind <- which((pdag == 1 & t(pdag) == 0), arr.ind = TRUE) ## a -> b
        for (i in seq_len(nrow(ind))) {
          a <- ind[i,1]
          b <- ind[i,2]
          indC <- which( (pdag[b,] == 1 & pdag[,b] == 1) & (pdag[a,] == 0 & pdag[,a] == 0))
          if (length(indC) > 0) {
            pdag[b,indC] <- 1
            pdag[indC,b] <- 0
            if (verbose)
              cat("\nRule 1:",a,"->",b," and ",b,"-",indC,
                  " where ",a," and ",indC," not connected: ",b,"->",indC,"\n")
          }
        }
        ## x11()
        ## plot(as(pdag,"graphNEL"), main="After Rule1")

        ## rule 2
        ind <- which((pdag == 1 & t(pdag) == 1), arr.ind = TRUE) ## a -> b
        for (i in seq_len(nrow(ind))) {
          a <- ind[i,1]
          b <- ind[i,2]
          indC <- which( (pdag[a,] == 1 & pdag[,a] == 0) & (pdag[,b] == 1 & pdag[b,] == 0))
          if (length(indC) > 0) {
            pdag[a,b] <- 1
            pdag[b,a] <- 0
            if (verbose) cat("\nRule 2: Kette ",a,"->",indC,"->",
                             b,":",a,"->",b,"\n")
          }
        }
        ## x11()
        ## plot(as(pdag,"graphNEL"), main="After Rule2")

        ## rule 3
        ind <- which((pdag == 1 & t(pdag) == 1), arr.ind = TRUE) ## a - b
        for (i in seq_len(nrow(ind))) {
          a <- ind[i,1]
          b <- ind[i,2]
          indC <- which( (pdag[a,] == 1 & pdag[,a] == 1) & (pdag[,b] == 1 & pdag[b,] == 0))
          if (length(indC) >= 2) {
            ## cat("R3: indC = ",indC,"\n")
            g2 <- pdag[indC,indC]
            ## print(g2)
            if (length(g2) <= 1) {
              g2 <- 0
            } else {
              diag(g2) <- rep(1,length(indC)) ## no self reference
            }
            if (any(g2 == 0)) { ## if two nodes in g2 are not connected
              pdag[a,b] <- 1
              pdag[b,a] <- 0
              if (verbose) cat("\nRule 3:",a,"->",b,"\n")
            }
          }
        }
        ## x11()
        ## plot(as(pdag,"graphNEL"), main="After Rule3")

        ## rule 4
        ##-         ind <- which((pdag==1 & t(pdag)==1), arr.ind=TRUE) ## a - b
        ##-         if (length(ind)>0) {
        ##-           for (i in seq_len(nrow(ind))) {
        ##-             a <- ind[i,1]
        ##-             b <- ind[i,2]
        ##-             indC <- which( (pdag[a,]==1 & pdag[,a]==1) & (pdag[,b]==0 & pdag[b,]==0))
        ##-             l.indC <- length(indC)
        ##-             if (l.indC>0) {
        ##-               found <- FALSE
        ##-               ic <- 0
        ##-               while(!found & (ic < l.indC)) {
        ##-                 ic <- ic + 1
        ##-                 c <- indC[ic]
        ##-                 indD <- which( (pdag[c,]==1 & pdag[,c]==0) & (pdag[,b]==1 & pdag[b,]==0))
        ##-                 if (length(indD)>0) {
        ##-                   found <- TRUE
        ##-                   pdag[b,a] = 0
        ##-                   if (verbose) cat("Rule 4 applied \n")
        ##-                 }
        ##-               }
        ##-             }
        ##-           }
        ##-         }

      }
      res@graph <- as(pdag,"graphNEL")
    } else {
      ## was not extendable; random DAG chosen
      res@graph <- res2$graph
      ## convert to CPDAG
      res@graph <- dag2cpdag(res@graph)
    }
  }
  return(res)
} ## udag2pdag


udag2pdagSpecial <- function(gInput, verbose = FALSE, n.max = 100) {
  ## Purpose: Transform the Skeleton of a pcAlgo-object to a PDAG using
  ## the rules of Pearl. The output is again a pcAlgo-object. Ambiguous
  ## v-structures are reoriented until extendable or max number of tries
  ## is reached. If still not extendable, a DAG is produced starting from the
  ## current PDAG even if introducing new v-structures.
  ##
  ## ----------------------------------------------------------------------
  ## Arguments:
  ## - gInput: pcAlgo object
  ## - verbose: 0 - no output, 1 - detailed output
  ## - n.max: Maximal number of tries to reorient v-strucutres
  ## ----------------------------------------------------------------------
  ## Values:
  ## - pcObj: Oriented pc-Object
  ## - evisit: Matrix counting the number of orientation attemps per edge
  ## - xtbl.orig: Is original graph with v-structure extendable
  ## - xtbl: Is final graph with v-structure extendable
  ## - amat0: Adj.matrix of original graph with v-structures
  ## - amat1: Adj.matrix of graph with v-structures after reorienting
  ##          edges from double edge visits
  ## - status:
  ##   0: original try is extendable
  ##   1: reorienting double edge visits helps
  ##   2: orig. try is not extendable; reorienting double visits don't help;
  ##      result is acyclic, has orig. v-structures, but perhaps
  ##      additional v-structures
  ## - counter: Number of reorientation tries until success or max.tries
  ## ----------------------------------------------------------------------
  ## Author: Markus Kalisch, Date: Sep 2006, 15:03
  counter <- 0
  res <- gInput
  status <- 0
  p <- length(nodes(res@graph))
  evisit <- amat0 <- amat1 <- matrix(0,p,p)
  xtbl <- xtbl.orig <- TRUE
  if (numEdges(gInput@graph) > 0) {
    g <- as(gInput@graph,"matrix") ## g_ij if i->j
    p <- dim(g)[1]
    pdag <- g
    ind <- which(g == 1,arr.ind = TRUE)
    ## ind <- unique(t(apply(ind,1,sort)))

    ## Create minimal pattern
    for (i in seq_len(nrow(ind))) {
      x <- ind[i,1]
      y <- ind[i,2]
      allZ <- setdiff(which(g[y,] == 1),x) ## x-y-z
      for(z in allZ) {
        if ((g[x,z] == 0) &&
            !(y %in% gInput@sepset[[x]][[z]] ||
              y %in% gInput@sepset[[z]][[x]])) {
          if (verbose) {
            cat("\n",x,"->",y,"<-",z,"\n")
            cat("Sxz=",gInput@sepset[[z]][[x]],"Szx=",gInput@sepset[[x]][[z]])
          }
          ## check if already in other direction directed
          if (pdag[x,y] == 0 && pdag[y,x] == 1) {
            evisit[x,y] <- evisit[x,y] + 1
            evisit[y,x] <- evisit[y,x] + 1
          }
          if (pdag[z,y] == 0 && pdag[y,z] == 1) {
            evisit[z,y] <- evisit[z,y] + 1
            evisit[y,z] <- evisit[y,z] + 1
          }
          pdag[x,y] <- pdag[z,y] <- 1
          pdag[y,x] <- pdag[y,z] <- 0
        } ## if
      } ## for
    } ## for ( i )

    amat0 <- pdag
    ## Test whether this pdag allows a consistent extension
    res2 <- pdag2dag(as(pdag,"graphNEL"))
    xtbl <- res2$success
    xtbl.orig <- xtbl

    if (!xtbl && (max(evisit) > 0)) {
      tmp.ind2 <- unique(which(evisit > 0,arr.ind = TRUE))
      ind2 <- unique(t(apply(tmp.ind2,1,sort)))
      ## print(ind2)
      n <- nrow(ind2)
      n.max <- min(2^n-1,n.max)
      counter <- 0
      ## xtbl is FALSE because of if condition
      while((counter < n.max) & !xtbl) {
        ## if (counter%%100 == 0) cat("\n counter=",counter,"\n")
        counter <- counter + 1
        dgBase <- digitsBase(counter)
        dgBase <- dgBase[length(dgBase):1]
        ## print(dgBase)
        indBase <- matrix(0,1,n)
        indBase[1,seq_along(dgBase)] <- dgBase
        ## indTmp <- ind2[ss[[counter]],,drop=FALSE]
        indTmp <- ind2[(indBase == 1),,drop = FALSE]
        ## print(indTmp)
        pdagTmp <- flipEdges(pdag,indTmp)
        resTmp <- pdag2dag(as(pdagTmp,"graphNEL"))
        xtbl <- resTmp$success
      }
      pdag <- pdagTmp
      status <- 1
    }
    amat1 <- pdag

    if (xtbl) {
      ## Convert to complete pattern: use rules by Pearl
      old_pdag <- matrix(0, p,p)
      while (any(old_pdag != pdag)) {
        old_pdag <- pdag
        ## rule 1
        ind <- which((pdag == 1 & t(pdag) == 0), arr.ind = TRUE) ## a -> b
        for (i in seq_len(nrow(ind))) {
            a <- ind[i,1]
            b <- ind[i,2]
            indC <- which( (pdag[b,] == 1 & pdag[,b] == 1) & (pdag[a,] == 0 & pdag[,a] == 0))
            if (length(indC) > 0) {
              pdag[b,indC] <- 1
              pdag[indC,b] <- 0
              if (verbose)
                cat("\nRule 1:",a,"->",b," and ",b,"-",indC,
                    " where ",a," and ",indC," not connected: ",b,"->",indC,"\n")
            }
        }
        ## x11()
        ## plot(as(pdag,"graphNEL"), main="After Rule1")

        ## rule 2
        ind <- which((pdag == 1 & t(pdag) == 1), arr.ind = TRUE) ## a -> b
        for (i in seq_len(nrow(ind))) {
            a <- ind[i,1]
            b <- ind[i,2]
            indC <- which( (pdag[a,] == 1 & pdag[,a] == 0) & (pdag[,b] == 1 & pdag[b,] == 0))
            if (length(indC) > 0) {
              pdag[a,b] <- 1
              pdag[b,a] <- 0
              if (verbose) cat("\nRule 2: Kette ",a,"->",indC,"->",
                    b,":",a,"->",b,"\n")
            }
        }
        ## x11()
        ## plot(as(pdag,"graphNEL"), main="After Rule2")

        ## rule 3
        ind <- which((pdag == 1 & t(pdag) == 1), arr.ind = TRUE) ## a - b
        for (i in seq_len(nrow(ind))) {
            a <- ind[i,1]
            b <- ind[i,2]
            indC <- which( (pdag[a,] == 1 & pdag[,a] == 1) & (pdag[,b] == 1 & pdag[b,] == 0))
            if (length(indC) >= 2) {
              ## cat("R3: indC = ",indC,"\n")
              g2 <- pdag[indC,indC]
              ## print(g2)
              if (length(g2) <= 1) {
                g2 <- 0
              } else {
                diag(g2) <- rep(1,length(indC)) ## no self reference
              }
              if (any(g2 == 0)) { ## if two nodes in g2 are not connected
                pdag[a,b] <- 1
                pdag[b,a] <- 0
                if (verbose) cat("\nRule 3:",a,"->",b,"\n")
              }
          }
        }
      }
      res@graph <- as(pdag,"graphNEL")
    } else {
      res@graph <- dag2cpdag(pdag2dag(as(pdag,"graphNEL"),keepVstruct = FALSE)$graph)
      status <- 2
      ## res@graph <- res2$graph
    }
  }
  list(pcObj = res, evisit = evisit, xtbl = xtbl, xtbl.orig = xtbl.orig,
       amat0 = amat0, amat1 = amat1, status = status, counter = counter)
}

udag2pdagRelaxed <- function(gInput, contextVars, verbose = FALSE, unfVect = NULL, solve.confl = FALSE, orientCollider = TRUE, rules = rep(TRUE, 3))
{

##################################################
  ## Internal functions
##################################################

  ## replace 'else if' branch in 'if( !solve.confl )' statement
  orientConflictCollider <- function(pdag, x, y, z) { ## x - y - z
    ## pdag: amat, pdag[x,y] = 1 and pdag[y,x] = 0 means x -> y
    ## x,y,z: colnumber of nodes in pdag
    ## only used if conflicts should be solved

    ## orient x - y
    if (pdag[x,y] == 1) {
      ## x --- y, x --> y => x --> y
      pdag[y,x] <- 0
    } else {
      ## x <-- y, x <-> y => x <-> y
      pdag[x,y] <- pdag[y,x] <- 2
    }

    ## orient z - y
    if (pdag[z,y] == 1) {
      ## z --- y, z --> y => z --> y
      pdag[y,z] <- 0
    } else {
      ## z <-- y, z <-> y => z <-> y
      pdag[z,y] <- pdag[y,z] <- 2
    }

    pdag
  }

  rule1 <- function(pdag, solve.confl = FALSE, unfVect = NULL) {
    ## Rule 1: a -> b - c goes to a -> b -> c
    ## Interpretation: No new collider is introduced
    ## Out: Updated pdag
    search.pdag <- pdag
    ind <- which(pdag == 1 & t(pdag) == 0, arr.ind = TRUE)
    for (i in seq_len(nrow(ind))) {
      a <- ind[i, 1]
      b <- ind[i, 2]
      ## find all undirected neighbours of b not adjacent to a
      isC <- which(search.pdag[b, ] == 1 & search.pdag[, b] == 1 &
                   search.pdag[a, ] == 0 & search.pdag[, a] == 0)
      if (length(isC) > 0) {
        for (ii in seq_along(isC)) {
          c <- isC[ii]
          ## if the edge between b and c has not been oriented previously,
          ## orient it using normal R1
          if (!solve.confl | (pdag[b,c] == 1 & pdag[c,b] == 1) ) { ## no conflict
            ## !! before, we checked search.pdag, not pdag !!
            if (!is.null(unfVect)) { ## deal with unfaithful triples
              if (!any(unfVect == triple2numb(p, a, b, c), na.rm = TRUE) &&
                  !any(unfVect == triple2numb(p, c, b, a), na.rm = TRUE)) {
                ## if unfaithful triple, don't orient
                pdag[b, c] <- 1
                pdag[c, b] <- 0
              }
            } else {
              ## don't care about unfaithful triples -> just orient
              pdag[b, c] <- 1
              pdag[c, b] <- 0
              ## cat("Rule 1\n")
            }
            if (verbose)
              cat("\nRule 1':", a, "->", b, " and ",
                  b, "-", c, " where ", a, " and ", c,
                  " not connected and ", a, b, c, " faithful triple: ",
                  b, "->", c, "\n")
          } else if (pdag[b,c] == 0 & pdag[c,b] == 1) {
            ## conflict that must be solved
            ## solve conflict: if the edge is b <- c because of a previous
            ## orientation within for loop then output <->
            if (!is.null(unfVect)) { ## deal with unfaithful triples
              if (!any(unfVect == triple2numb(p, a, b, c), na.rm = TRUE) &&
                  !any(unfVect == triple2numb(p, c, b, a), na.rm = TRUE)) {
                pdag[b, c] <- 2
                pdag[c, b] <- 2
                if (verbose)
                  cat("\nRule 1':", a, "->", b, "<-",
                      c, " but ", b, "->", c, "also possible and",
                      a, b, c, " faithful triple: ", a,"->", b, "<->", c,"\n")
              }
            } else {
              ## don't care about unfaithful triples -> just orient
              pdag[b, c] <- 2
              pdag[c, b] <- 2
              if (verbose)
                cat("\nRule 1':", a, "->", b, "<-",
                    c, " but ", b, "->", c, "also possible and",
                    a, b, c, " faithful triple: ", a,"->", b, "<->", c,"\n")
            } ## unfVect: if else
          } ## conflict: if else
        } ## for c
      } ## if length(isC)
      if (!solve.confl) search.pdag <- pdag
    } ## for ind
    pdag
  }

  rule2 <- function(pdag, solve.confl = FALSE) {
    ## Rule 2: a -> c -> b with a - b: a -> b
    ## Interpretation: Avoid cycle
    ## normal version = conservative version
    search.pdag <- pdag
    ind <- which(search.pdag == 1 & t(search.pdag) == 1, arr.ind = TRUE)
    for (i in seq_len(nrow(ind))) {
      a <- ind[i, 1]
      b <- ind[i, 2]
      isC <- which(search.pdag[a, ] == 1 & search.pdag[, a] == 0 &
                   search.pdag[, b] == 1 & search.pdag[b, ] == 0)
      for (ii in seq_along(isC)) {
        c <- isC[ii]
        ## if the edge has not been oriented yet, orient it with R2
        ## always do this if you don't care about conflicts
        if (!solve.confl | (pdag[a, b] == 1 & pdag[b, a] == 1) ) {
          pdag[a, b] <- 1
          pdag[b, a] <- 0
          if (verbose)
            cat("\nRule 2: Chain ", a, "->", c,
                "->", b, ":", a, "->", b, "\n")
        }
        ## else if the edge has been oriented as a <- b by a previous R2
        else if (pdag[a, b] == 0 & pdag[b, a] == 1) {
          pdag[a, b] <- 2
          pdag[b, a] <- 2
          if (verbose)
            cat("\nRule 2: Chain ", a, "->", c,
                "->", b, ":", a, "<->", b, "\n")
        }
      }
      if (!solve.confl) search.pdag <- pdag
    }
    pdag
  }

  rule3 <- function(pdag, solve.confl = FALSE, unfVect = NULL) {
    ## Rule 3: a-b, a-c1, a-c2, c1->b, c2->b but c1 and c2 not connected;
    ## then a-b => a -> b
    search.pdag <- pdag
    ind <- which(search.pdag == 1 & t(search.pdag) == 1, arr.ind = TRUE)
    for (i in seq_len(nrow(ind))) {
      a <- ind[i, 1]
      b <- ind[i, 2]
      c <- which(search.pdag[a, ] == 1 & search.pdag[, a] == 1 &
                 search.pdag[, b] == 1 & search.pdag[b, ] == 0)
      if (length(c) >= 2) {
        cmb.C <- combn(c, 2)
        cC1 <- cmb.C[1, ]
        cC2 <- cmb.C[2, ]
        for (j in seq_along(cC1)) {
          c1 <- cC1[j]
          c2 <- cC2[j]
          if (search.pdag[c1, c2] == 0 && search.pdag[c2,c1] == 0) {
            if (!is.null(unfVect)) {
              if (!any(unfVect == triple2numb(p, c1, a, c2), na.rm = TRUE) &&
                  !any(unfVect == triple2numb(p, c2, a, c1), na.rm = TRUE)) {
                ## if the edge has not been oriented yet, orient it with R3
                if (!solve.confl | (pdag[a, b] == 1 & pdag[b, a] == 1) ) {
                  pdag[a, b] <- 1
                  pdag[b, a] <- 0
                  if (!solve.confl) search.pdag <- pdag
                  if (verbose)
                    cat("\nRule 3':", a, c1, c2, "faithful triple: ",
                        a, "->", b, "\n")
                  break
                }
                ## else if: we care about conflicts and  the edge has been oriented as a <- b by a previous R3
                else if (pdag[a, b] == 0 & pdag[b, a] == 1) {
                  pdag[a, b] <- pdag[b, a] <- 2
                  if (verbose)
                    cat("\nRule 3':", a, c1, c2, "faithful triple: ",
                        a, "<->", b, "\n")
                  break
                } ## if solve conflict
              } ## if unf. triple found
            } else { ## if care about unf. triples; else don't care
              if (!solve.confl | (pdag[a, b] == 1 & pdag[b, a] == 1) ) {
                pdag[a, b] <- 1
                pdag[b, a] <- 0
                if (!solve.confl) search.pdag <- pdag
                if (verbose)
                  cat("\nRule 3':", a, c1, c2, "faithful triple: ",
                      a, "->", b, "\n")
                break
              }
              ## else if: we care about conflicts and  the edge has been oriented as a <- b by a previous R3
              else if (pdag[a, b] == 0 & pdag[b, a] == 1) {
                pdag[a, b] <- pdag[b, a] <- 2
                if (verbose)
                  cat("\nRule 3':", a, c1, c2, "faithful triple: ",
                      a, "<->", b, "\n")
                break
              } ## if solve conflict
            } ## if care about unf. triples
          } ## if c1 and c2 are not adjecent
        } ## for all pairs of c's
      } ## if at least two c's are found
    } ## for all undirected edges
    pdag
  }

##################################################
  ## Main
##################################################

  ## prepare adjacency matrix of skeleton
  if (numEdges(gInput@graph) == 0)
    return(gInput)
  g <- as(gInput@graph, "matrix")
  p <- nrow(g)
  pdag <- g

  ## orient collider
  if (orientCollider) {
    ind <- which(g == 1, arr.ind = TRUE)
    for (i in seq_len(nrow(ind))) {
      x <- ind[i, 1]
      y <- ind[i, 2]
      allZ <- setdiff(which(g[y, ] == 1), x) ## x - y - z
      for (z in allZ) {
        ## check collider condition
        if (g[x, z] == 0 &&
            !((y %in% gInput@sepset[[x]][[z]]) ||
              (y %in% gInput@sepset[[z]][[x]]))) {
          if (length(unfVect) == 0) { ## no unfaithful triples
            if (!solve.confl) { ## don't solve conflicts
              # added: make sure y is not the collider
              if (!(y %in% contextVars)){
                pdag[x, y] <- pdag[z, y] <- 1
                pdag[y, x] <- pdag[y, z] <- 0
              }else{
                 if(!(x %in% contextVars)){
                    pdag[y, x] <- 1
                    pdag[x, y] <- 0
                 }
                 if(!(z %in% contextVars)){
                    pdag[y, z] <- 1
                    pdag[z, y] <- 0
                 }
              }
            } else { ## solve conflicts
              pdag <- orientConflictCollider(pdag, x, y, z)
            }
          } else { ## unfaithful triples are present
            if (!any(unfVect == triple2numb(p, x, y, z), na.rm = TRUE) &&
                !any(unfVect == triple2numb(p, z, y, x), na.rm = TRUE)) {
              if (!solve.confl) { ## don't solve conflicts
                if (!(y %in% contextVars)){
                  pdag[x, y] <- pdag[z, y] <- 1
                  pdag[y, x] <- pdag[y, z] <- 0
                }else{
                   if(!(x %in% contextVars)){
                      pdag[y, x] <- 1
                      pdag[x, y] <- 0
                   }
                   if(!(z %in% contextVars)){
                      pdag[y, z] <- 1
                      pdag[z, y] <- 0
                   }
                }
              } else { ## solve conflicts
                pdag <- orientConflictCollider(pdag, x, y, z)
              }
            }
          }
        }
      } ## for z
    } ## for i
  } ## end: Orient collider

  ## Rules 1 - 3
  repeat {
    old_pdag <- pdag
    if (rules[1]) {
      pdag <- rule1(pdag, solve.confl = solve.confl, unfVect = unfVect)
    }
    if (rules[2]) {
      pdag <- rule2(pdag, solve.confl = solve.confl)
    }
    if (rules[3]) {
      pdag <- rule3(pdag, solve.confl = solve.confl, unfVect = unfVect)
    }
    if (all(pdag == old_pdag))
      break
  } ## repeat

  gInput@graph <- as(pdag, "graphNEL")
  gInput
}
