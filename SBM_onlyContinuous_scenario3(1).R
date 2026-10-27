library('igraph')
library('mvtnorm')
source('FitAttribute.R')
source('FitAttribute_chooseK.R')
library(rlist)
library(progress)


gen_scenario3 <- function(seed) {
  set.seed(seed)
  n <- 300
  z <- rep(1:4, each = 75)                      # ground-truth blocks
  
  ## -- undirected Bernoulli SBM, no self-loops --
  theta <- matrix(c(0.5,   0.5,   0.005, 0.005,
                    0.5,   0.5,   0.005, 0.005,
                    0.005, 0.005, 0.5,   0.5,
                    0.005, 0.005, 0.5,   0.5), 4, 4, byrow = TRUE)
  A  <- matrix(0L, n, n)
  ij <- which(upper.tri(A), arr.ind = TRUE)
  p  <- theta[cbind(z[ij[, 1]], z[ij[, 2]])]
  e  <- rbinom(nrow(ij), 1L, p)
  A[ij]        <- e
  A[ij[, 2:1]] <- e
  A  <- Matrix::Matrix(A, sparse = TRUE)
  
  ## -- 2-D Gaussian attribute (common Sigma across blocks) --
  ##    blocks 1 & 4 sit low; blocks 2 & 3 sit high  -> {1,4} vs {2,3}
  mu  <- list(c(0.4, -0.5), c(2.6, 2.5), c(3.4, 2.5), c(-0.4, -0.5))
  Sig <- matrix(c(0.05, 0.001, 0.001, 0.05), 2)
  Xg  <- matrix(NA_real_, n, 2)
  for (k in 1:4) {
    idx <- which(z == k)
    Xg[idx, ] <- MASS::mvrnorm(length(idx), mu[[k]], Sig)
  }
  
  ## -- two multinomial attributes, both coding the {1,4} vs {2,3} split --
  ## X_c1: 3 cats ; X_c2: 4 cats.  Rows indexed by block k = 1..4
  P1 <- rbind(c(0.90, 0.05, 0.05),          # block 1
              c(0.05, 0.90, 0.05),          # block 2
              c(0.05, 0.90, 0.05),          # block 3
              c(0.90, 0.05, 0.05))          # block 4
  P2 <- rbind(c(0.85, 0.05, 0.05, 0.05),    # block 1
              c(0.05, 0.85, 0.05, 0.05),    # block 2
              c(0.05, 0.85, 0.05, 0.05),    # block 3
              c(0.85, 0.05, 0.05, 0.05))    # block 4
  draw <- function(row) sample.int(length(row), 1L, prob = row)
  xc1 <- vapply(z, function(k) draw(P1[k, ]), integer(1))
  xc2 <- vapply(z, function(k) draw(P2[k, ]), integer(1))
  Xc  <- data.frame(c1 = factor(xc1, levels = 1:3),
                    c2 = factor(xc2, levels = 1:4))
  
  list(A = A, Xg = Xg, Xc = Xc, z = z)
}

z3 <- rep(1:4, each = 75) 
aris <- list()
aris_k <- list()

pb <- progress_bar$new(total = 500)

for (sed in 1:100){
  starttime<-Sys.time()
  pb$tick()
  
  dat3 <- gen_scenario3(seed = sed)
  X <- cbind(dat3$Xg, model.matrix(~ c1 + c2, data = dat3$Xc)[, -1])
  colnames(X) <- c("cont1", "cont2", "c1_2", "c1_3", "c2_2", "c2_3", "c2_4")
  Out=FitAttribute(dat3$A,X,0)
  Out_k=FitAttribute_k(dat3$A,X,0,4)
  ari <- mclust::adjustedRandIndex(Out$Comm, z3)
  ari_k <- mclust::adjustedRandIndex(Out_k$Comm, z3)
  
  aris <- append(aris, ari)
  aris_k <- append(aris_k, ari_k)
  list.save(aris, 'aris3.rdata')
  list.save(aris_k, 'aris3_k.rdata')
  print(Sys.time()-starttime)
}




arisxx   <- list.load('aris3_k.rdata')

