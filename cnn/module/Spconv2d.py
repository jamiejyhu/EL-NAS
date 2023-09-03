class Spconv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,*args,**kwargs):
        super(Spconv2d,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
            )
    def forward(self,x):
        n,c,h,w = x.size()
        assert c % 4 == 0
        x1 = x[:,:c//4,:,:]
        x2 = x[:,c//4:c//2,:,:]
        x3 = x[:,c//2:c//4*3,:,:]
        x4 = x[:,c//4*3:c,:,:]
        x1 = nn.functional.pad(x1,(1,0,1,0),mode = "constant",value = 0) # left top
        x2 = nn.functional.pad(x2,(0,1,1,0),mode = "constant",value = 0) # right top
        x3 = nn.functional.pad(x3,(1,0,0,1),mode = "constant",value = 0) # left bottom
        x4 = nn.functional.pad(x4,(0,1,0,1),mode = "constant",value = 0) # right bottom
        x = torch.cat([x1,x2,x3,x4],dim = 1)
        return self.conv(x)
