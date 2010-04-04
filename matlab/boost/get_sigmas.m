function sigma_list  = get_sigmas(sigma_list, max_sigma,k)

if sigma_list(length(sigma_list))*k < max_sigma
    sigma_list = [sigma_list sigma_list(length(sigma_list))*k];
    %keyboard;
    sigma_list = get_sigmas(sigma_list, max_sigma, k);
end