library('igraph')
library('mvtnorm')
source('FitAttribute.R')
source('FitAttribute_chooseK.R')
library(rlist)
library(progress)


gen_scenario1 <- function(seed) {
  set.seed(seed)
  n <- 150
  z <- rep(1:3, each = 50)                      # ground-truth blocks
  
  ## -- undirected Bernoulli SBM, no self-loops --
  theta <- matrix(c(0.4, 0.2, 0.1,
                    0.2, 0.4, 0.2,
                    0.1, 0.2, 0.4), 3, 3, byrow = TRUE)
  A  <- matrix(0L, n, n)
  ij <- which(upper.tri(A), arr.ind = TRUE)     # all i < j pairs
  p  <- theta[cbind(z[ij[, 1]], z[ij[, 2]])]
  e  <- rbinom(nrow(ij), 1L, p)
  A[ij]        <- e
  A[ij[, 2:1]] <- e                             # symmetrise (lower triangle)
  A  <- Matrix::Matrix(A, sparse = TRUE)        # greed-friendly format
  
  ## -- 2-D Gaussian attribute, one (mu_k, Sigma_k) per block --
  mu  <- list(c(0.5, 0.5), c(2.5, 2.5), c(-1.5, 2.5))
  Sig <- list(matrix(c(0.09, 0.001, 0.001, 0.09), 2),
              matrix(c(0.08, 0.002, 0.002, 0.09), 2),
              matrix(c(0.09, 0.003, 0.003, 0.08), 2))
  Xg <- matrix(NA_real_, n, 2)
  for (k in 1:3) {
    idx <- which(z == k)
    Xg[idx, ] <- MASS::mvrnorm(length(idx), mu[[k]], Sig[[k]])
  }
  
  ## -- two multinomial attributes: X_c1 (3 cats), X_c2 (4 cats) --
  P1 <- rbind(c(0.80, 0.15, 0.05),
              c(0.15, 0.80, 0.05),
              c(0.05, 0.15, 0.80))
  P2 <- rbind(c(0.70, 0.15, 0.05, 0.10),
              c(0.15, 0.70, 0.05, 0.10),
              c(0.05, 0.15, 0.70, 0.10))
  draw <- function(row) sample.int(length(row), 1L, prob = row)
  xc1 <- vapply(z, function(k) draw(P1[k, ]), integer(1))
  xc2 <- vapply(z, function(k) draw(P2[k, ]), integer(1))
  # LCA expects a data.frame of FACTORS; fix levels so every replication
  # carries all categories even if one happens to be unsampled.
  Xc <- data.frame(c1 = factor(xc1, levels = 1:3),
                   c2 = factor(xc2, levels = 1:4))
  
  list(A = A, Xg = Xg, Xc = Xc, z = z)
}

z1 <- rep(1:3, each = 50)                      # ground-truth blocks
aris <- list()
aris_k <- list()

pb <- progress_bar$new(total = 500)
for (sed in 1:500){
  pb$tick()
  
dat1 <- gen_scenario1(seed = sed)
X <- cbind(dat1$Xg, model.matrix(~ c1 + c2, data = dat1$Xc)[, -1])
colnames(X) <- c("cont1", "cont2", "c1_2", "c1_3", "c2_2", "c2_3", "c2_4")

Out=FitAttribute(dat1$A,X,0)
ari <- mclust::adjustedRandIndex(Out$Comm, z1)
aris <- append(aris, ari)

Out_k=FitAttribute_k(dat1$A,X,0,3)
ari_k <- mclust::adjustedRandIndex(Out_k$Comm, z1)
aris_k <- append(aris_k, ari_k)

list.save(aris, 'aris1.rdata')
list.save(aris_k, 'aris1_k.rdata')
}




arisxx   <- list.load('aris1_k.rdata')
