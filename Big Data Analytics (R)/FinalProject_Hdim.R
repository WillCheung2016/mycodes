#require(corrplot)
#require(fields)

data_pts<-read.table("d:/myrworks/dim512.txt",header=FALSE)
data_pts<-data.matrix(data_pts)
colnames(data_pts)<-NULL

data_pts<-t(data_pts)
num_pts<-1024

indexes<-sample.int(num_pts,num_pts)
data_pts<-data_pts[,indexes]

true_centers<-read.table("d:/myrworks/dim512_ans.txt",header=FALSE)
true_centers<-data.matrix(true_centers)
colnames(true_centers)<-NULL

true_centers<-t(true_centers)

find_nearest<-function(pt,centers,num_clusters){
  dist<-rep(0,num_clusters)
  for (i in c(1:num_clusters)){
    dist[i]<-norm(pt-centers[,i],'2')
  }
  closest<-which(dist==min(dist))
  return(closest)
}

compute_dist<-function(cluster_centers,true_centers,num_clusters){
  temp<-0
  cluster_count<-num_clusters
  for (i in c(1:(num_clusters-1))){

    closest<-find_nearest(cluster_centers[,1],true_centers,cluster_count)
    
    temp<-temp + norm(cluster_centers[,1]-true_centers[,closest],'2')
    cluster_centers<-cluster_centers[,-1]
    true_centers<-true_centers[,-closest]
    cluster_count<-cluster_count-1
  }
  temp<-temp + norm(cluster_centers-true_centers,'2')
  return(temp)
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

compute_harmonic_weights<-function(pt,centers,dims,num_clusters){
  weights<-rep(0,num_clusters)
  difference<-matrix(rep(pt,num_clusters),nrow=dims,ncol=num_clusters)-centers
  dif_norm_vec<-sqrt(colSums(difference*difference))
  recp_sum<-sum(1/(dif_norm_vec**2))**2
  for (i in c(1:num_clusters)){
    center_dist<-dif_norm_vec[i]
    weights[i]<-(2*num_clusters)/((center_dist**4)*recp_sum)
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

strat_sampling<-function(pts,sample_size,centers,num_clusters,cluster_counts,dims){
  ratios<-cluster_counts/sum(cluster_counts)
  accumulator<-matrix(rep(0,dims*num_clusters),nrow=dims,ncol=num_clusters)
  counters<-rep(0,num_clusters)
  for (i in c(1:sample_size)){
    membership<-find_nearest(pts[,i],centers,num_clusters)
    accumulator[,membership]<-accumulator[,membership]+pts[,i]
    counters[membership]<-counters[membership]+1
  }
  temp<-matrix(rep(0,dims),nrow=dims)
  for (j in c(1:num_clusters)){
    if (counters[j]!=0){
      t<-accumulator[,j]/counters[j]
      temp<-temp+ratios[j]*t
      #temp<-temp+ratios[j]*centers[,j]
    }
    
  }
  return(temp)
}

strat_sampling_cluster<-function(centers,num_clusters,cluster_counts,dims){
  ratios<-cluster_counts/sum(cluster_counts)
  temp<-matrix(rep(0,dims),nrow=dims)
  for (j in c(1:num_clusters)){
    temp<-temp+ratios[j]*centers[,j]
  }
  
  return(temp)
}

mixDensity<-function(pt,centers,covariances,num_clusters,cluster_counts){
  ratios<-cluster_counts/sum(cluster_counts)
  temp<-0
  for (i in c(1:num_clusters)){
    temp<-temp+ratios[i]*dmvnorm(pt,mean=centers[,i],sigma=covariances[,,i])
  }
  return(temp)
}

locate_closest<-function(samples,num_samples,centers,num_clusters){
  t<-rep(0,num_samples)
  for (i in c(1:num_samples)){
    t[i]<-find_nearest(samples[,i],centers,num_clusters)
  }
  return(t)
}

KL_dist_unif<-function(tags,num_clusters){
  temp<-0
  pu<-1/num_clusters
  for (i in c(1:num_clusters)){
    temp<-temp+(pu)*log(pu/(length(tags[tags==i])/length(tags)))
  }
  return(temp)
}

KL_dist_prior<-function(pri_distr,sample_distr,num_clusters){
  temp<-0
  for (i in c(1:num_clusters)){
    if(length(tags[tags==i])!=0){
      temp<-temp+(pri_distr[i])*log(pri_distr[i]/sample_distr[i])
    }
  }
  return(temp)
}

C_mat<-function(num_clusters,constant=0.5){
  #temp<-matrix(1,num_clusters,num_clusters)-diag(num_clusters)
  temp<-matrix(1,num_clusters,num_clusters)
  return(constant*temp)
}

num_steps<-6000
num_clusters<-16
dims<-512
#initial_steps<-ceiling(log(0.05)/log(1-(1/num_clusters)))
initial_steps<-100
current_pointer<-initial_steps+1
current_centers<-data_pts[,sample.int(initial_steps,num_clusters)]
#current_centers<-data_pts[,c(1:num_clusters)]
current_weights<-rep(1,num_clusters)
prior_sample_size<-10
samples<-array(rep(0,dims*prior_sample_size*num_clusters),c(dims,prior_sample_size,num_clusters))
sample_pointer<-rep(1,num_clusters)
sample_cov<-array(rep(0,dims*dims*num_clusters),c(dims,dims,num_clusters))
cluster_counts<-rep(0,num_clusters)
alpha<-0.7
gamma<-0
# Online K-means Starts
C<-C_mat(num_clusters,constant=1)
vals<-c()

for (i in c(1:num_steps)){
  vals<-c(vals,compute_dist(current_centers,true_centers,num_clusters))
  closest<-find_nearest(data_pts[,current_pointer],current_centers,num_clusters)
  weights_vec<-compute_weights(data_pts[,current_pointer],current_centers,closest,n=1,p=-1,dims=dims, num_clusters=num_clusters)
  #weights_vec<-compute_harmonic_weights(data_pts[,current_pointer],current_centers,dims=dims, num_clusters=num_clusters)
  #ada<-1/((current_pointer+2)**alpha)
  ada<-0.005
  for (j in c(1:num_clusters)){
    #current_centers[,j]<-current_centers[,j]-ada*(-1)*weights_vec[j]*(as.matrix(data_pts[,current_pointer])-current_centers[,j])-2*gamma*current_centers%*%C[,j]
    current_centers[,j]<-current_centers[,j]-ada*weights_vec[j]*(as.matrix(data_pts[,current_pointer])-current_centers[,j])-2*gamma*current_centers%*%C[,j]
    
  }
  current_pointer<-current_pointer+1
  if (current_pointer==1025){
    current_pointer<-1
  }
  samples[,sample_pointer[closest],closest]<-data_pts[,current_pointer]
  sample_pointer[closest]<-sample_pointer[closest]+1
  if (sample_pointer[closest]==(prior_sample_size+1)){
    sample_pointer[closest]<-1
  }
  
  if (i>(num_steps-3000)){
    cluster_counts[closest]<-cluster_counts[closest]+1
  }
}

for (i in c(1:num_clusters)){
  repCenter<-matrix(rep(current_centers[,i],prior_sample_size),c(dims,prior_sample_size))
  diff_mat<-samples[,,i]-repCenter
  sample_cov[,,i]<-diff_mat%*%t(diff_mat)/(prior_sample_size-1)
}

#Dist<-rdist(t(current_centers),t(true_centers))
#corrplot(Dist,is.corr=FALSE)

###################Stratified Sampling########################################

pop_center<-rowSums(data_pts)/num_pts
sample_size<-100
strat_buffer<-matrix(rep(0,dims*sample_size),c(dims,sample_size))
strat_pointer<-1
num_samplingSteps<-2000
strat_dist<-c()
naive_dist<-c()
if (current_pointer==1025){
  current_pointer<-1
}
#cluster_counts<-rep(1,num_clusters)
for (i in c(1:num_samplingSteps)){
  
  strat_buffer[,strat_pointer]<-data_pts[,current_pointer]
  current_pointer<-current_pointer+1
  strat_pointer<-strat_pointer+1
  #print(strat_pointer)
  if (strat_pointer==(sample_size+1)){
    est_center<-strat_sampling(strat_buffer,sample_size,current_centers,num_clusters,cluster_counts,dims)
    naive_center<-rowSums(strat_buffer)/sample_size
    strat_dist<-c(strat_dist,norm(pop_center-est_center,'2'))
    naive_dist<-c(naive_dist,norm(pop_center-naive_center,'2'))
    strat_pointer<-1
    strat_buffer<-matrix(rep(0,dims*sample_size),c(dims,sample_size))
  }
  if (current_pointer==1025){
    current_pointer<-1
  }
}
est_center<-strat_sampling(strat_buffer,sample_size,current_centers,num_clusters,cluster_counts,dims)
naive_center<-rowSums(strat_buffer)/sample_size

strat_dist<-c(strat_dist,norm(pop_center-est_center,'2'))
naive_dist<-c(naive_dist,norm(pop_center-naive_center,'2'))
