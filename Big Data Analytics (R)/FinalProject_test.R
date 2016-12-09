load("D:/myrworks/proj_data_v2.RData")
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
  temp<-matrix(c(0,0),nrow=2)
  for (j in c(1:num_clusters)){
    if (counters[j]!=0){
      t<-accumulator[,j]/counters[j]
      temp<-temp+ratios[j]*t
      #temp<-temp+ratios[j]*centers[,j]
    }
    
  }
  return(temp)
}

strat_sampling_cluster<-function(centers,num_clusters,cluster_counts){
  ratios<-cluster_counts/sum(cluster_counts)
  temp<-matrix(c(0,0),nrow=2)
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

logLH<-function(pts,n_pts,centers,covariances,num_clusters,cluster_counts){
  temp<-0
  ratios<-cluster_counts/sum(cluster_counts)
  for (i in c(1:n_pts)){
    closest<-find_nearest(pts[,i],centers,num_clusters)
    t<-mixDensity(pts[,i],centers,covariances,num_clusters,cluster_counts)
    temp<-temp+log(ratios[closest])+log(t)
  }
  return(temp)
}

membership_list<-function(mem_vec,num_clusters){
  temp_list<-list()
  for (i in c(1:num_clusters)){
    temp_list[[i]]<-c(0)
  }
  #print(temp_list)
  for (i in c(1:length(mem_vec))){
    temp<-temp_list[[mem_vec[i]]]
    temp_list[[mem_vec[i]]]<-c(temp,i)
    #print(lappend(temp_list[mem_vec[i]],list(i)))
  }
  for (i in c(1:num_clusters)){
    temp_list[[i]]<-temp_list[[i]][-1]
  }
  return(temp_list)
}

count_samples<-function(sample_centers,num_clusters){
  temp<-rep(0,num_clusters)
  for(i in c(1:length(sample_centers))){
    temp[sample_centers[i]]<-temp[sample_centers[i]]+1
  }
  return(temp)
}

num_steps<-1000
initial_steps<-ceiling(log(0.05)/log(1-(1/num_clusters)))
current_pointer<-initial_steps+1
#current_centers<-data_pts[,sample.int(initial_steps,num_clusters)]
current_centers<-data_pts[,c(1:num_clusters)]
current_weights<-rep(1,num_clusters)
prior_sample_size<-250
samples<-array(rep(0,dims*prior_sample_size*num_clusters),c(dims,prior_sample_size,num_clusters))
sample_pointer<-rep(1,num_clusters)
sample_cov<-array(rep(0,dims*dims*num_clusters),c(dims,dims,num_clusters))
cluster_counts<-rep(0,num_clusters)
alpha<-0.55
# Online K-means Starts

plot(data_pts[1,1:current_pointer],data_pts[2,1:current_pointer],col='green',main="Initialization of IWK",xlab="x-axis",ylab="y-axis")
points(current_centers[1,],current_centers[2,],pch=24,bg='red')
legend("topleft", inset=0.05, title="Data",c("Points","Centers"),pch=c(1,24))

for (i in c(1:num_steps)){
  closest<-find_nearest(data_pts[,current_pointer],current_centers,num_clusters)
  #weights_vec<-compute_weights(data_pts[,current_pointer],current_centers,closest,n=1,p=-1,dims=dims, num_clusters=num_clusters)
  weights_vec<-compute_harmonic_weights(data_pts[,current_pointer],current_centers,dims=dims, num_clusters=num_clusters)
  ada<-1/((current_pointer+2)**alpha)
  for (j in c(1:num_clusters)){
    current_centers[,j]<-current_centers[,j]+ada*weights_vec[j]*(as.matrix(data_pts[,current_pointer])-current_centers[,j])
    #current_centers[,j]<-current_centers[,j]-ada*weights_vec[j]*(as.matrix(data_pts[,current_pointer])-current_centers[,j])
    
  }
  current_pointer<-current_pointer+1
  samples[,sample_pointer[closest],closest]<-data_pts[,current_pointer]
  sample_pointer[closest]<-sample_pointer[closest]+1
  if (sample_pointer[closest]==(prior_sample_size+1)){
    sample_pointer[closest]<-1
  }
  if (i>(num_steps-100)){
    cluster_counts[closest]<-cluster_counts[closest]+1
  }
}

for (i in c(1:num_clusters)){
  repCenter<-matrix(rep(current_centers[,i],prior_sample_size),c(dims,prior_sample_size))
  diff_mat<-samples[,,i]-repCenter
  sample_cov[,,i]<-diff_mat%*%t(diff_mat)/(prior_sample_size-1)
}

plot(data_pts[1,1:current_pointer],data_pts[2,1:current_pointer],col='green',main="Clusters and Centers",xlab="x-axis",ylab="y-axis")
points(current_centers[1,],current_centers[2,],pch=24,bg='red')
legend("topleft", inset=0.05, title="Data",c("Points","Centers"),pch=c(1,24))
#LH_vec<-c(LH_vec,logLH(data_pts,num_pts,current_centers,sample_cov,num_clusters,cluster_counts))

print(current_centers)

# Online Expectation Maximization Starts

minBatch_size<-100
num_EMsteps<-10000-current_pointer+1
alpha<-1
buffer<-matrix(rep(0,dims*minBatch_size),c(dims,minBatch_size))
buffer_pointer<-1
for (i in c(1:num_EMsteps)){
  buffer[,buffer_pointer]<-data_pts[,current_pointer]
  current_pointer<-current_pointer+1
  buffer_pointer<-buffer_pointer+1
  if ((i%%minBatch_size)==0){
    H<-matrix(rep(0,num_clusters*minBatch_size),c(minBatch_size,num_clusters))
    H1<-matrix(rep(0,num_clusters*minBatch_size),c(minBatch_size,num_clusters))
    for (j in c(1:minBatch_size)){
      closest<-find_nearest(buffer[,j],current_centers,num_clusters)
      H[j,]<-softIndices(buffer[,j],sample_cov,current_centers,cluster_counts,num_clusters)
      H1[j,closest]<-1
    }
    ada<-1/((current_pointer+2)**alpha)
    D<-diag(colSums(H))
    current_centers<-(1-ada)*current_centers+ada*buffer%*%H%*%solve(D)
    #print(current_centers)
    update_cov<-cov_updates(buffer,minBatch_size,current_centers,H,dims,num_clusters)
    for (k in c(1:num_clusters)){
      sample_cov[,,k]<-(1-ada)*sample_cov[,,k]+ada*update_cov[,,k]
    }
    buffer_pointer<-1
    cluster_counts<-cluster_counts+colSums(H1)
    #stepSize<-stepSize+1
    #LH_vec<-c(LH_vec,logLH(data_pts,num_pts,current_centers,sample_cov,num_clusters,cluster_counts))
    #plot(data_pts[1,1:current_pointer],data_pts[2,1:current_pointer],col='green',main="Clusters and Centers",xlab="x-axis",ylab="y-axis")
    #points(current_centers[1,],current_centers[2,],pch=24,bg='red')
  }
  
  
}
plot(data_pts[1,1:current_pointer],data_pts[2,1:current_pointer],col='green',main="Clusters and Centers",xlab="x-axis",ylab="y-axis")
points(current_centers[1,],current_centers[2,],pch=24,bg='red')
legend("topleft", inset=0.05, title="Data",c("Points","Centers"),pch=c(1,24))
print(current_centers)
#LH_vec<-c(LH_vec,logLH(data_pts,num_pts,current_centers,sample_cov,num_clusters,cluster_counts))


###################################################################################

# Stratified Sampling or pseudo-MCMC Sampling Starts here

mcmc_size<-100

#mcmc_samples<-matrix(rep(0,dims*mcmc_size),c(dims,mcmc_size))
mcmc_samples<-data_pts[,current_pointer:(current_pointer+mcmc_size-1)]
pt_old<-data_pts[,(current_pointer-1)]
current_pointer<-current_pointer+mcmc_size
ratios<-cluster_counts/sum(cluster_counts)
mcmc_sample_centers<-locate_closest(mcmc_samples,mcmc_size,current_centers,num_clusters)
pos_list<-membership_list(mcmc_sample_centers,num_clusters)
sample_ratios<-count_samples(mcmc_sample_centers,num_clusters)
sample_ratios<-sample_ratios/sum(sample_ratios)
#print(pos_list)
MC_KLdist<-c()
KLdist<-c()
count<-0

while(current_pointer!=(num_pts+1)){
  if (count%%100==0){
    MC_KLdist<-c(MC_KLdist,KL_dist_prior(ratios,sample_ratios,num_clusters))
    #MC_KLdist<-c(MC_KLdist,KL_dist_unif(mcmc_sample_centers,num_clusters))
    #buffer<-data_pts[,(current_pointer-mcmc_size):(current_pointer-1)]
    #buffer_centers<-locate_closest(buffer,mcmc_size,current_centers,num_clusters)
    #KLdist<-c(KLdist,KL_dist_unif(buffer_centers,num_clusters))
  }
  pt_new<-data_pts[,current_pointer]
  old_cluster<-find_nearest(pt_old,current_centers,num_clusters)
  new_cluster<-find_nearest(pt_new,current_centers,num_clusters)
  #accept<-mixDensity(pt_old,current_centers,sample_cov,num_clusters,cluster_counts)/mixDensity(pt_new,current_centers,sample_cov,num_clusters,cluster_counts)
  accept<-ratios[old_cluster]/ratio[new_cluster]
  #print(accept)
  u<-runif(1)
  if (u<accept){
    sample_ratios<-count_samples(mcmc_sample_centers,num_clusters)
    sample_ratios<-sample_ratios/sum(sample_ratios)
    print(sample_ratios)
    cluster<-sample.int(num_clusters,1,prob=sample_ratios)
    while(sample_ratios[cluster]<=ratios[cluster]){
      cluster<-sample.int(num_clusters,1,prob=sample_ratios)
    }
    #pos<-sample.int(length(pos_list[[cluster]]),1)
    ind<-pos_list[[cluster]][1]
    mcmc_samples[,ind]<-pt_new
    pos_list[[cluster]]<-pos_list[[cluster]][-1]
    pos_list[[new_cluster]]<-c(pos_list[[new_cluster]],ind)
    pt_old<-pt_new
    mcmc_sample_centers[ind]<-new_cluster
    #print(c(cluster,new_cluster))
  }
  current_pointer<-current_pointer+1
  count<-count+1
}


hist(MC_KLdist,breaks=20,main="Histogram of KL Divergence Data Set 1",xlab="KL Distances")
###########################################################################
