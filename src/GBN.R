
library(bnlearn)

get_index <- function(i, j, k, n_col) {
  return(3*i*n_col + 3*j + k + 2)
}

mydata <- read.csv("/Users/bill/Documents/PGMProject/src/data/resized_data_scaled_32.csv")
#mydata[] <- lapply( mydata, factor)
mydata$y <- as.factor(mydata$y)

col_names <- colnames(mydata)
nvars <- ncol(mydata)
n_rows <- sqrt((nvars-1)/3)
n_cols <- sqrt((nvars-1)/3)


dag = empty.graph(col_names)

adj = matrix(0L, ncol = nvars, nrow = nvars,
             dimnames = list(col_names, col_names))

# simple naive bayes
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


amat(dag) = adj

nparams(dag, mydata)
#graphviz.plot(dag)

set.seed(42)

K <- 3

train_score <- c(0,0,0)
test_score <- c(0,0,0)
fit_time <- c(0,0,0)
score_time <- c(0,0,0)

for (i in 1:K) {
  ## 75% of the sample size
  train_size <- floor(0.90 * nrow(mydata))
  test_size <- nrow(mydata) - train_size
  
  train_ind <- sample(seq_len(nrow(mydata)), size = train_size)
  
  train <- mydata[train_ind, ]
  test <- mydata[-train_ind, ]
  
  print("FIT")
  start_time <- proc.time()
  fit <- bn.fit(dag, train)
  fit_time[i] <- (proc.time() - start_time)[[3]]
  
  print("Pred")
  train_pred <- predict(fit, "y", train, 'bayes-lw')
  
  start_time <- proc.time()
  test_pred <- predict(fit, "y", test, 'bayes-lw')
  score_time[i] <- (proc.time() - start_time)[[3]]
  print("Pred")
  
  test_score[i] <- sum(test[['y']] == test_pred, na.rm = TRUE) / test_size
  train_score[i] <- sum(train[['y']] == train_pred, na.rm = TRUE) / train_size
  print("Done")
  print(c(fit_time[i], score_time[i], train_score[i], test_score[i]))
}

mean_train_score <- mean(train_score)
mean_test_score <- mean(test_score)
mean_fit_time <- mean(fit_time)
mean_score_time <- mean(score_time)

std_train_score <- sd(train_score)
std_test_score <- sd(test_score)
std_fit_time <- sd(fit_time)
std_score_time <- sd(score_time)

print(c(mean_fit_time, std_fit_time, mean_score_time, std_score_time, mean_train_score, std_train_score, mean_test_score, std_test_score))

