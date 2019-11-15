data {
  int<lower = 0> nsite;
  int<lower = 0> nyear;
  int<lower = 0> nrep;
  int<lower = -1, upper = nrep> Y[nsite, nyear]; // -1 acts as NA/missing
  int<lower = 1> n_l1;
  int<lower = 1, upper = n_l1> l1[nsite];

  int<lower = 0> n_pred;
  int<lower = 1, upper = n_l1> l1_pred[n_pred];

  int<lower = 0, upper = 1> any_surveys[nsite, nyear];
  
  // fixed effect design matrices include an intercept column
  int<lower = 1> m;
  matrix[nsite, m] X;
  matrix[n_pred, m] X_pred;
  
  int<lower = 1> m_p;
  matrix[nsite, m_p] X_p[nyear];
  matrix[n_pred, m_p] X_p_pred[nyear];
}

transformed data {
  matrix[nsite, nyear] zero_obs;
  matrix[nsite, nyear] nonzero_obs;
  matrix[nsite, nyear] log_zero_obs;
  
  for (i in 1:nsite) {
    for (t in 1:nyear) {
      zero_obs[i, t] = Y[i, t] == 0;
      nonzero_obs[i, t] = (Y[i, t] != 0) * 1e-7;
      log_zero_obs[i, t] = log(zero_obs[i, t] + nonzero_obs[i, t]);
    }
  }
}
parameters {
  // the ecoregion adjustments are 4 dimensional:
  // dim 1: intercept adj on detection probs
  // dim 2: intercept adj on colonization
  // dim 3: intercept adj on persistence
  // dim 4: intercept adj on initial occupancy
  vector[4] alpha; // global terms
  matrix[m, 4] beta;
  cholesky_factor_corr[4] L_l1;
  matrix[n_l1, 4] z_l1;
  vector<lower = 0>[4] sigma_l1;
  vector[m_p] beta_p; // fixed effects for detection covariates
}

transformed parameters {
  matrix<lower = 0, upper = 1>[nsite, nyear] p;
  vector<lower = 0, upper = 1>[nsite] gamma;
  vector<lower = 0, upper = 1>[nsite] phi;
  vector<lower = 0, upper = 1>[nsite] psi1;
  matrix[n_l1, 4] l1_adj;
  matrix[nsite, 4] adj;
  simplex[2] ps[2, nsite];

  l1_adj = (diag_pre_multiply(sigma_l1, L_l1) *  z_l1')';

  for (i in 1:4) {
    adj[, i] = X * beta[, i] + l1_adj[l1, i];
  }
  
  for (t in 1:nyear) {
    p[, t] = inv_logit(adj[, 1] + X_p[t] * beta_p);
  }
  gamma = inv_logit(adj[, 2]);
  phi = inv_logit(adj[, 3]);
  psi1 = inv_logit(adj[, 4]);
  
  // fill in elements of transition matrix
  for (i in 1:nsite) {
    ps[1, i, 1] = phi[i];
    ps[1, i, 2] = 1 - phi[i];
    ps[2, i, 1] = gamma[i];
    ps[2, i, 2] = 1 - gamma[i];
  }
}

model {
  // priors
  alpha ~ std_normal();
  to_vector(beta) ~ std_normal();

  to_vector(z_l1) ~ std_normal();
  L_l1 ~ lkj_corr_cholesky(10);
  sigma_l1 ~ gamma(1.5, 10);

  beta_p ~ std_normal();

  // Likelihood
  for (i in 1:nsite) {
    real acc[2];
    vector[2] gam[nyear];

    // First year
    if (any_surveys[i, 1]) {
      gam[1, 1] = log(psi1[i]) + binomial_lpmf(Y[i, 1] | nrep, p[i, 1]);
      gam[1, 2] = log1m(psi1[i]) + log_zero_obs[i, 1];
    } else {
      gam[1, 1] = log(psi1[i]);
      gam[1, 2] = log1m(psi1[i]);
    }

    for (t in 2:nyear) {
      if (any_surveys[i, t]) {
        for (k in 1:2) {
          for (j in 1:2)
            if (k == 1) {
              acc[j] = gam[t - 1, j] + log(ps[j, t - 1, k]) + binomial_lpmf(Y[i, t] | nrep, p[i, t]);
            } else {
              acc[j] = gam[t - 1, j] + log(ps[j, t - 1, k]) + log_zero_obs[i, t];
            }
          gam[t, k] = log_sum_exp(acc);
        }
      } else {
        // no surveys: substitute identity matrix for po 
        // (equivalent to leaving it out)
        for (k in 1:2) {
          for (j in 1:2)
            acc[j] = gam[t - 1, j] + log(ps[j, t - 1, k]);
          gam[t, k] = log_sum_exp(acc);
        }
      }
    }
    target += log_sum_exp(gam[nyear]);
  }
}

generated quantities {
  matrix<lower = 0, upper = 1>[n_pred, nyear] p_pred;
  vector<lower = 0, upper = 1>[n_pred] gamma_pred;
  vector<lower = 0, upper = 1>[n_pred] phi_pred;
  vector<lower = 0, upper = 1>[n_pred] psi1_pred;
  matrix[n_pred, 4] adj_pred;
  
  for (i in 1:4) {
    adj_pred[, i] = X_pred * beta[, i] + l1_adj[l1_pred, i];
  }
  
  for (t in 1:nyear) {
    p_pred[, t] = inv_logit(adj_pred[, 1] + X_p_pred[t] * beta_p);
  }
  
  gamma_pred = inv_logit(adj_pred[, 2]);    
  phi_pred = inv_logit(adj_pred[, 3]);
  psi1_pred = inv_logit(adj_pred[, 4]);
}
