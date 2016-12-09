load("D:/myrworks/proj_data.RData")
require(mvtnorm)

find_nearest<-function(pt,centers,num_clusters){
  dist<-rep(0,num_clusters)
  for (i in c(1:num_clusters)){
    dist[i]<-norm(pt-centers[,i],'2')
  }
  closest<-which(dist==min(dist))
  return(closest)
}

compute_weights<-function(pt,centers,closest,n,p,dims,num_clusters){
  weights<-rep(0,num_clusters)
  difference<-matrix(rep(pt,num_clusters),nrow=dims,ncol=num_clusters)-centers
  dif_norm_vec<-sqrt(colSums(difference*difference))
  if (which(dif_norm_vec==min(dif_norm_vec))!=closest){
    print("error")
    return(0)
  }
  for (i in c(1:num_clusters)){
    if (i == closest){
      remain_rec<-1/(dif_norm_vec**p)
      remain_rec[i]<-0
      diff_norm<-dif_norm_vec[i]
      weights[i]<-(-1)*(n-p)*(diff_norm**(n-p-2))-n*(diff_norm**(n-2))*sum(remain_rec)
      
    }
    else{
      closest_diff<-dif_norm_vec[closest]
      weights[i]<-p*(closest_diff**n)/(dif_norm_vec[i]**(p+2))
    }
  }
  return(weights)
}

softIndices<-function(pt,cov_mat,centers,counts,num_clusters){
  LH<-rep(0,num_clusters)
  ratios<-counts/sum(counts)
  for (i in c(1:num_clusters)){
    difference<-pt-centers[,i]
    inv_cov<-solve(cov_mat[,,i])
    LH[i]<-ratios[i]*sqrt(det(inv_cov))*exp((-1/2)*t(difference)%*%inv_cov%*%difference)
  }
  LH<-LH/sum(LH)
  return(LH)
}

cov_updates<-function(buffer,buffer_size,centers,H,dims,num_clusters){
  updates<-array(rep(0,dims*dims*num_clusters),c(dims,dims,num_clusters))
  sumH<-colSums(H)
  for (i in c(1:num_clusters)){
    repCenter<-matrix(rep(centers[,i],buffer_size),c(dims,buffer_size))
    D<-diag(H[,i]/sumH[i])
    diff<-buffer-repCenter
    updates[,,i]<-diff%*%D%*%t(diff)
  }
  return(updates)
}

num_steps<-200
initial_steps<-ceiling(log(0.05)/log(1-(1/num_clusters)))
current_pointer<-initial_steps+1
current_centers<-data_pts[,sample.int(initial_steps,num_clusters)]
current_weights<-rep(1,num_clusters)
prior_sample_size<-5
samples<-array(rep(0,dims*prior_sample_size*num_clusters),c(dims,prior_sample_size,num_clusters))
sample_pointer<-rep(1,num_clusters)
sample_cov<-array(rep(0,dims*dims*num_clusters),c(dims,dims,num_clusters))
cluster_counts<-rep(0,num_clusters)
alpha<-0.8
# Online K-means Starts

for (i in c(1:num_steps)){
  closest<-find_nearest(data_pts[,current_pointer],current_centers,num_clusters)
  weights_vec<-compute_weights(data_pts[,current_pointer],current_centers,closest,n=1,p=-1,dims=dims, num_clusters=num_clusters)
  print(current_centers)
  for (j in c(1:num_clusters)){
    current_centers[,j]<-current_weights[j]*current_centers[,j]+weights_vec[j]*as.matrix(data_pts[,current_pointer])
    current_centers[,j]<-current_centers[,j]/(current_weights[j]+weights_vec[j])
    current_weights[j]<-current_weights[j]+weights_vec[j]
    
  }
  current_pointer<-current_pointer+1
  samples[,sample_pointer[closest],closest]<-data_pts[,current_pointer]
  sample_pointer[closest]<-sample_pointer[closest]+1
  if (sample_pointer[closest]==(prior_sample_size+1)){
    sample_pointer[closest]<-1
  }
  cluster_counts[closest]<-cluster_counts[closest]+1
}

for (i in c(1:num_clusters)){
  sample_cov[,,i]<-cov(t(samples[,,i]))
}

plot(data_pts[1,1:current_pointer],data_pts[2,1:current_pointer],col='green',main="Clusters and Centers",xlab="x-axis",ylab="y-axis")
points(current_centers[1,],current_centers[2,],pch=24,bg='red')

# Online Expectation Maximization Starts



