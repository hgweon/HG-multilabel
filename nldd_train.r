
## Nearest Labelsets Using Double Distances 
## This function trains a NLDD model for multi-label classification.
## Currently, Support Vector Machines (SVM) can only be used as 
## the base classifier.
## To run the nldd_train function, the e1071 package is required.
## An output of the function is a list that includes 
## the binomial regression model and binary SVM models.

install.packages("e1071") # for using SVM as base classifier


## data: data.frame containing the data features and labels
## label_index: vector containing the indices of labels
## feature_scale: if TRUE, standardized features are used
## for calculating distances in the feature spaces

nldd_train <- function(data,label_index,feature_scale=TRUE)
{ 
  y_length <- length(label_index)
  data2 <- cbind(data[,label_index],data[,-label_index])
  n1 <- nrow(data2)
  n2 <- round(nrow(data2)/2)
  a <- sample(n1,round(n2),replace=FALSE)  
  T1 <- data2[a,]
  T2 <- data2[-a,]
  pred_M <- matrix(rep(0,nrow(T2)*y_length),ncol=y_length)
  bin_model <- list(1)
  for(i in 1:y_length)
  {
  	if(sum(data2[,i]) > 0)
  	{
      tr <- cbind(y=as.factor(data2[,i]),data2[,-c(1:y_length)])  		
	    bin_model[[i]] <- svm(y~.,data=tr,probability=TRUE,kernel="linear",scale=FALSE)
	}
    if(sum(T1[,i]) > 0)
    {
      tr <- cbind(y=as.factor(T1[,i]),T1[,-c(1:y_length)])
      ts <- cbind(y=as.factor(T2[,i]),T2[,-c(1:y_length)])
      m <- svm(y~.,data=tr,probability=TRUE,kernel="linear",scale=FALSE)
      pred <- predict(m,ts,probability=TRUE)
      temp <- as.data.frame(attr(pred,"probabilities"))
      a <- which(names(temp)=="1")
      pred_M[,i] <- attr(pred,"probabilities")[,a]       
      }
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

  S <- c(0,0,0)
	ind <- 1:nrow(T1)
	T <- as.matrix(T1[,1:y_length])
	for(i in 1:nrow(T2))	#Obtain set S
	{
  		temp1 <- t(t(T)-pred_M[i,])
  		dy <- sqrt(apply(temp1*temp1,1,sum))
  		temp2 <- t(t(X_tr2)-X_ts2[i,])
  		dx <- sqrt(apply(temp2*temp2,1,sum))
  		dist <- cbind(dx,dy)

  		a <- which(dy==min(dy))
  		a2 <- which(dx[a]==min(dx[a]))
  		indy <- a[sort(a2)[1]]
  		a <- which(dx==min(dx))
  		a2 <- which(dy[a]==min(dy[a]))
  		indx <- a[sort(a2)[1]]
  		if(indy!=indx) ind <- c(indy,indx)
  		if(indy==indx) ind <- indy

  		S2 <- dist[ind,]
  		temp <- t(t(T[ind,])==as.numeric(T2[i,1:y_length]))
  		if(length(ind)==1) loss <-   y_length-sum(temp)
 		  if(length(ind)>1) loss <- (y_length-apply(temp,1,sum))  
      if(length(ind)>1) S2 <- cbind(S2,loss)
      if(length(ind)==1) S2 <- c(S2,loss)
  		S <- rbind(S,S2)
	}
	S <- as.data.frame(S[-1,])
	non_dup <- which(duplicated(S)!=1)
	S <- S[non_dup,]
	names(S) <- c("dx","dy","loss")
	out <- glm(cbind(loss,rep(y_length,nrow(S)))~.,data=S,family=binomial)
	return(list(binom_est = out,BR_model = bin_model))
}

