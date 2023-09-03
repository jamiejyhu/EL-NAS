class Octconv(nn.Module): 
    def __init__(self, ch_in, ch_out, kernel_size, stride=1, alphas=[0.5,0.5], padding=0): 
        super(Octconv, self).__init__()

        # get layer parameters 
        self.alpha_in, self.alpha_out = alphas
        assert 0 <= self.alpha_in <= 1 and 0 <= self.alpha_out <= 1, "Alphas must be in interval [0, 1]"
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding## (kernel_size - stride ) // 2 padding
        
        # Calculate the exact number of high/low frequency channels 
        self.ch_in_lf = int(self.alpha_in*ch_in)
        self.ch_in_hf = ch_in - self.ch_in_lf
        self.ch_out_lf = int(self.alpha_out*ch_out)
        self.ch_out_hf = ch_out - self.ch_out_lf

        # Create convolutional and other modules necessary
        self.hasLtoL = self.hasLtoH = self.hasHtoL = self.hasHtoH = False
        if (self.ch_in_lf and self.ch_out_lf):    
            self.hasLtoL = True
            self.conv_LtoL = nn.Sequential(nn.Conv2d(self.ch_in_lf, self.ch_out_lf, self.kernel_size, padding=self.padding,bias=False),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.BatchNorm2d(self.ch_out_lf, 0.8),
                                           )
        if (self.ch_in_lf and self.ch_out_hf): 
            self.hasLtoH = True
            self.conv_LtoH = nn.Sequential(nn.Conv2d(self.ch_in_lf, self.ch_out_hf, self.kernel_size, padding=self.padding,bias=False),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.BatchNorm2d(self.ch_out_hf, 0.8),
                                           )
        if (self.ch_in_hf and self.ch_out_lf):
            self.hasHtoL = True
            self.conv_HtoL = nn.Sequential(nn.Conv2d(self.ch_in_hf, self.ch_out_lf, self.kernel_size, padding=self.padding,bias=False),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.BatchNorm2d(self.ch_out_lf, 0.8),
                                           )
        if (self.ch_in_hf and self.ch_out_hf):
            self.hasHtoH = True
            self.conv_HtoH = nn.Sequential(nn.Conv2d(self.ch_in_hf, self.ch_out_hf, self.kernel_size, padding=self.padding,bias=False),
                                           nn.LeakyReLU(0.2, inplace=True),
                                           nn.BatchNorm2d(self.ch_out_hf, 0.8),
                                           )
        self.avg_pool  = nn.AvgPool2d(2,2)
        
    def forward(self, input): 
        
        # Split input into high frequency and low frequency components
        fmap_w = input.shape[-1]
        fmap_h = input.shape[-2]
        # We resize the high freqency components to the same size as the low frequency component when 
        # sending out as output. So when bringing in as input, we want to reshape it to have the original  
        # size as the intended high frequnecy channel (if any high frequency component is available). 
        input_hf = input

        input_hf = input[:,:self.ch_in_hf,:,:]
        input_lf = input[:,self.ch_in_hf:,:,:]
        input_lf = self.avg_pool(input_lf)
        
        # Create all conditional branches 
        LtoH = HtoH = LtoL = HtoL = 0.
        if (self.hasLtoL):
            LtoL = self.conv_LtoL(input_lf)
        if (self.hasHtoH):
            HtoH = self.conv_HtoH(input_hf)
            op_h, op_w = HtoH.shape[-2], HtoH.shape[-1]
            HtoH = HtoH.reshape(-1, self.ch_out_hf, op_h, op_w)
        if (self.hasLtoH):
            LtoH = F.interpolate(self.conv_LtoH(input_lf), scale_factor=2.25, mode='bilinear')#(7*1.0)/3
            op_h, op_w = LtoH.shape[-2], LtoH.shape[-1]
            LtoH = LtoH.reshape(-1, self.ch_out_hf, op_h, op_w)
        if (self.hasHtoL):
            HtoL = self.avg_pool(self.conv_HtoL(input_hf))
        
        # Elementwise addition of high and low freq branches to get the output
        out_hf = LtoH + HtoH
        out_lf = LtoL + HtoL

        out_hf = self.avg_pool(out_hf)
        
        if (self.ch_out_lf == 0):
            return out_hf
        if (self.ch_out_hf == 0):
            return out_lf
        op = torch.cat([out_hf,out_lf],dim=1)
        return op
