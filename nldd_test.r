## Nearest Labelsets Using Double Distances 
## This function predicts the labelset of a multi-label instance
## using the NLDD method.

## train_data: training data (data.frame)
## test_data: test data (data.frame)
## nldd_object: output of the nldd_train function
## label_index:  vector containing the indices of labels
## (must be the same for both train_data and test_data)
## feature_scale: if TRUE, standardized features are used
## for calculating distances in the feature spaces

nldd_test <- function(train_data,test_data,nldd_object,label_index,feature_scale=TRUE)
{ 
	y_length <- length(label_index)
	T1 <- cbind(train_data[,label_index],train_data[,-label_index])
	T2 <- cbind(test_data[,label_index],test_data[,-label_index])
	pred_M <- matrix(rep(0,nrow(T2)*y_length),ncol=y_length)
  	for(i in 1:y_length)
  	{
  		m <- (nldd_object$BR_model)[[i]]
  		pred <- predict(m,T2,probability=TRUE)
    	temp <- as.data.frame(attr(pred,"probabilities"))
    	a <- which(names(temp)=="1")
    	pred_M[,i] <- attr(pred,"probabilities")[,a]
    }
    X_tr <- X_tr2 <- as.matrix(T1[,-c(1:y_length)])
	X_ts <- X_ts2 <- as.matrix(T2[,-c(1:y_length)])
	if(feature_scale=="TRUE") #Standardization
	{
  		nc <- ncol(X_tr)
  		for(i in 1:nc)
  		{
    		temp.mean <- mean(X_tr[,i])
    		temp.sd <- sd(X_tr[,i])
    		X_tr2[,i] <- (X_tr[,i]-temp.mean)/temp.sd
    		X_ts2[,i] <- (X_ts[,i]-temp.mean)/temp.sd
	    }
	}
	train_M <- as.matrix(T1[,1:y_length])
	pred_y <- matrix(0,nrow=nrow(X_ts2),ncol=y_length)
	for(i in 1:nrow(X_ts2))
	{
  		temp1 <- t(t(train_M)-pred_M[i,])
  		dy <- sqrt(apply(temp1*temp1,1,sum))
  		temp2 <- t(t(X_tr2)-X_ts2[i,])
  		dx <- sqrt(apply(temp2*temp2,1,sum))
  		dist <- cbind(dx,dy)
  		temp <- as.data.frame(dist)
  		A <- predict(nldd_object$binom_est,newdata=temp,type="response")
  		a <- which.min(A)
  		pred_y[i,] <- train_M[a,]
	}
	return(pred_y)
}
