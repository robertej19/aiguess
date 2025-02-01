class ImageProcessingParams:
    def __init__(self, ip_config):

        # if an ip_config was passed, process it:
        if ip_config is None:
            self.hf_noise_gaussian_kernel = [3, 3]
            self.hf_noise_gaussian_sigma  = 1
            self.sky_gaussian_kernel      = [125, 125]
            self.sky_gaussian_sigma       = 125
            self.hsv_lower_bound         = [0, 0,0]
            self.hsv_upper_bound         = [180, 255, 255]
            self.adaptive_threshold_max_value = 255
            self.adaptive_threshold_blockSize  = 5
            self.adaptive_threshold_constant   = 4
            self.sobel_pre_gaussian_kernel = [3,3]
            self.sobel_pre_gaussian_sigma  = 0.5
            self.sobel_x_kernel = 3
            self.sobel_y_kernel = 3
            self.sobel_threshold = 50
            self.object_area_threshold = 400000000000 
        else:
            self.hf_noise_gaussian_kernel = tuple(ip_config["hf_noise_gaussian_kernel"])
            self.hf_noise_gaussian_sigma  = ip_config["hf_noise_gaussian_sigma"]
            self.sky_gaussian_kernel      = tuple(ip_config["sky_gaussian_kernel"])
            self.sky_gaussian_sigma       = ip_config["sky_gaussian_sigma"]
            self.hsv_lower_bound         = tuple(ip_config["hsv_lower_bound"])
            self.hsv_upper_bound         = tuple(ip_config["hsv_upper_bound"])
            self.adaptive_threshold_max_value = ip_config["adaptive_threshold_max_value"]
            self.adaptive_threshold_blockSize  = ip_config["adaptive_threshold_block_size"]
            self.adaptive_threshold_constant   = ip_config["adaptive_threshold_constant"]
            self.sobel_pre_gaussian_kernel = tuple(ip_config["sobel_pre_gaussian_kernel"])
            self.sobel_pre_gaussian_sigma  = ip_config["sobel_pre_gaussian_sigma"]
            self.sobel_x_kernel = ip_config["sobel_x_kernel"]
            self.sobel_y_kernel = ip_config["sobel_y_kernel"]
            self.sobel_threshold = ip_config["sobel_threshold"]
            self.object_area_threshold = ip_config["object_area_threshold"]


