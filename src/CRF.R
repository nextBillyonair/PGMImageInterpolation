
library(CRF)

get_index <- function(i, j, k, n_col) {
  return(3*i*n_col + 3*j + k + 2)
}


mydata <- read.csv("/Users/bill/Documents/PGMProject/src/data/resized_data_1.csv")
mydata[] <- lapply( mydata, as.integer)
#mydata$y <- as.factor(mydata$y)

col_names <- colnames(mydata)
nvars <- ncol(mydata)
n_rows <- sqrt((nvars-1)/3)
n_cols <- sqrt((nvars-1)/3)

adj = matrix(0L, ncol = nvars, nrow = nvars,
             dimnames = list(col_names, col_names))


for (i in 0:(n_rows-1)) {
  for (j in 0:(n_cols-1)) {
    for (k in 0:2) {
      idx <- get_index(i,j,k,n_cols)
      adj[col_names[1], col_names[idx]] = 1L
      if (k==0){
        adj[col_names[idx], col_names[get_index(i,j, 1, n_cols)]] = 1L
        adj[col_names[idx], col_names[get_index(i,j, 2, n_cols)]] = 1L
      }
      if (k==1) {
        adj[col_names[idx], col_names[get_index(i,j, 2, n_cols)]] = 1L
      }
      if (i+1 < n_rows) {
        adj[col_names[idx], col_names[get_index(i+1,j, k, n_cols)]] = 1L
      }
      if (j+1 < n_cols) {
        adj[col_names[idx], col_names[get_index(i,j+1, k, n_cols)]] = 1L
      }
    }
  }
}



crf <- make.mrf(adj)
crf <- make.features(crf)
crf <- make.par(crf, 4)

start_time <- proc.time()
train.mrf(crf, mydata, node.fea = 1, edge.fea = 1)
end_time <- proc.time() - start_time
