

import torch, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from bonedataset import VoxelTransform as vt

# ============= General stuff =============#

# sample noise in hypersphere 
def hypersphere(z, radius=1):
    return z * radius / z.norm(p=2, dim=1, keepdim=True)

# distance between z-vectors 
def z_angle(z):
    z2 = z.view(z.shape[:2])
    idx = torch.arange(len(z)).expand(len(z),len(z))
    angle = torch.arccos(z2 @ z2.T)[(idx-idx.T)>0]
    return angle  

def fuzzymax(x, alpha = 20, dim = None):
    return torch.sum(x * 2**(alpha*x), dim = dim) / torch.sum(2**(alpha*x), dim = dim)
    
def fuzzymin(x, alpha = 20, dim = None):
    return fuzzymax(x, -alpha, dim)

# fast box sum 
def boxsum(patch, footprint):
    ps = patch.shape
    if footprint == ps[2]:
        x = torch.sum(patch.view(*ps[:2],-1),dim = 2).view(*ps[:2],1,1,1)
    else:
        one_ = torch.ones(1,1,1,1,1, device = patch.device)
        x = torch.conv3d(patch, one_.expand(ps[1],1,footprint,1,1), groups = ps[1])
        x = torch.conv3d(x, one_.expand(ps[1],1,1,footprint,1), groups = ps[1])
        x = torch.conv3d(x, one_.expand(ps[1],1,1,1,footprint), groups = ps[1])
    return x

# structural similarity
def ssim(patch, footprint = None, alpha = [1.,1.,1.], c1 = 0.01**2, c2 = 0.03**2,
         agg = torch.mean):
    ps = patch.shape
    assert ps[1]>1, 'patch must have at least two channels shape(patch)[1]>1'
    assert ps[2]*ps[3]*ps[4]>1, 'patch must have a volume > 1'
    idx = torch.arange(ps[1]).expand(ps[1],ps[1])
    mask = (idx-idx.T)>0
    x = idx[mask]
    y = idx.T[mask]
    footprint = ps[2] if footprint is None else min(ps[2], footprint)
    n = footprint**3
    avg = boxsum(patch, footprint)  / n       
    var = (boxsum(patch**2, footprint) - avg**2 * n) / (n - 1)
    var = torch.where(var < 0, torch.zeros_like(var), var)  # Evita negativos
    sd = (torch.clamp(var, min=1e-8)).sqrt()
    cov = torch.clip((boxsum(patch[:,x] * patch[:,y], footprint) - avg[:,x] * avg[:,y] * n) / (n-1),0,1)
    l_ = (2 * avg[:,x] * avg[:,y] + c1) / (avg[:,x]**2 + avg[:,y]**2 + c1)
    c_ = (2 * sd[:,x] * sd[:,y] + c2) / (sd[:,x]**2 + sd[:,y]**2 + c2)
    s_ = torch.abs((cov + c2/2) / (sd[:,x] * sd[:,y] + c2/2) )
    ssim_ = l_**alpha[0] * c_**alpha[1] * s_**alpha[2]
    if agg is not None: ssim_ = agg(ssim_.view(*ssim_.shape[:2],-1),dim = 2)
    return ssim_

def ceil(x):
    return int(x - int(x+1)) + int(x+1)

def _resample(patch, scale, sampler):
    for i in range(scale): patch = sampler(patch)
    return patch

def upsample(patch, scale = 0):    
    return _resample(patch, int(scale), torch.nn.Upsample(scale_factor = 2, mode = 'trilinear'))

# if scale is floting point number, downsampled patch is weighted average 
def downsample(patch, scale = 0): 
    downsampler = torch.nn.AvgPool3d(kernel_size = 2, stride = 2)
    patch = _resample(patch, int(scale), downsampler)
    alpha = scale % 1
    if alpha:
        upsampler = torch.nn.Upsample(scale_factor = 2, mode = 'trilinear')
        patch = alpha * upsampler(downsampler(patch.clone())) + (1 - alpha) * patch
    return patch
    
# ============= Routines of training losses =============#

# penalizing high SSIM if z angles are not similar  
# forces diversity of patches  
class SSIMPenalty:
    def __init__(self, min_angle = 0.1, alpha = [1.0,1.0,1.0], c1 = 0.01**2, 
                 c2 = 0.03**2,angle_pot = 1/10, footprint = None):
        self.c1 = c1
        self.c2 = c2
        self.min_angle = min_angle
        self.angle_pot = angle_pot
        self.alpha = alpha
        self.angle_dist = lambda angle: torch.clip((angle-self.min_angle)/
                (torch.pi-self.min_angle),0,1)**self.angle_pot
        self.angle_dist0 = self.angle_dist(torch.tensor(torch.pi/2))
        self.footprint = footprint
        
    def __call__(self, patch, z = None):
        angle_dist = self.angle_dist0.to(patch.device) if z is None else self.angle_dist(z_angle(z))
        eps = 0.001
        ssim_ = ssim(patch.swapaxes(0,1), self.footprint, self.alpha, self.c1, self.c2).flatten()
        error = (angle_dist + eps) / (1 + eps - ssim_) 
        return error

# class handling gradient penalty to organize weights of critic
class GradientPenalty:
    def __init__(self, D):
        self.D = D

    def __call__(self, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1, 1, requires_grad=True, device=real_data.device)
        patch = alpha * real_data + (1 - alpha) * fake_data
        score = self.D(patch)
        gradients = torch.autograd.grad(outputs=score, inputs=patch,
                                        grad_outputs=torch.ones_like(score),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(real_data.shape[0], -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

# class handling the progressive training pipeline

# ============= Routines to show training progress =============#
# 3d appearance of bone patch
class Raytracing:
    def __init__(self, t=150, s=50, colormap=matplotlib.cm.bone, bg = (0.,0.4,0.4,1.0), 
                    scale_vol=2, scale_im=2, mag_fac = 0.0, device = 'cpu'):
        self.upsample_vol = torch.nn.Upsample(scale_factor = scale_vol, mode = 'trilinear') if scale_vol !=1 else lambda x: x
        self.upsample_im = torch.nn.Upsample(scale_factor = scale_im, mode = 'bilinear') if scale_im !=1 else lambda x: x
        
        self.bg = torch.tensor(bg)
        self.colormap = lambda x: torch.tensor(colormap(torch.clip(x.cpu(),0,1)),device = device)
        self.binarize = lambda x: torch.clip(x-t+s,0,2*s)/(2*s)
        self.mask = None
        if mag_fac>0:
            maskz = torch.tensor([1.,0,-1.], device = device).view(1,1,3,1,1).expand(1,1,3,3,3)/9 * mag_fac
            self.mask = torch.cat([maskz,maskz.swapaxes(2,3),maskz.swapaxes(2,4)],dim = 0)    
    
    def __call__(self, patch):
        patchb = self.binarize(self.upsample_vol(patch))
        patchc = self.colormap(patchb).swapaxes(1,5)
        patchc = torch.nn.functional.pad(patchc.squeeze(5),(0,0,0,0,0,1))
        patchc[:,:,-1] = self.bg.view(1,4,1,1).expand(*patchc.shape[:2],*patchc.shape[3:])
        grad = 0
        if self.mask is not None:
            d2 = torch.conv3d(patchb, self.mask)**2
            grad = (torch.sum(d2[:,1:],dim=1,keepdims = True)**0.5 - d2[:,:1]**0.5)
            grad = torch.nn.functional.pad(grad,(1,1,1,1,1,1),value = 0)
        patchb = torch.nn.functional.pad(torch.clip( 1 - patchb - grad,0,1),(0,0,0,0,1,1),value = 1)
        patchb[:,:,-1] = (1-self.bg[3])
        patchb = torch.cumprod(patchb, 2)
        patchb = (patchb[:,:,:-1] - patchb[:,:,1:])
        im = torch.sum(patchc * patchb, dim = 2)
        im = torch.clip(self.upsample_im(im),0,1)
        im = im.unsqueeze(4).swapaxes(1,4)
        return im


# save screenshots of current epoch
class SaveBatch:
    def __init__(self, p_out, dpi = 300, t = 135, s = 35, mag_fac = 0.15, ext = '.png', device = 'cpu'):
        self.device = device
        self.rt = Raytracing(t = t, s = s, mag_fac = mag_fac, scale_im = 4, device = device)
        p_imagesf = os.path.join(p_out, 'fakeimages')
        if not os.path.exists(p_imagesf): os.makedirs(p_imagesf)
        self.cfnf = lambda fn: os.path.join(p_imagesf, fn)
        p_imagesr = os.path.join(p_out, 'realimages')
        if not os.path.exists(p_imagesr): os.makedirs(p_imagesr)
        self.cfnr = lambda fn: os.path.join(p_imagesr, fn)
        
        self.upsample_im = torch.nn.Upsample(scale_factor = 4, mode = 'bilinear')
        self.dpi = dpi
        self.ext = ext
        
    def __call__(self, patch, epoch = 0, kind = 'fake'):
        ps = patch.shape[2]
        fn = kind+'_epoch-'+str(epoch).zfill(3)+self.ext
        title = kind+' data: epoch='+str(epoch)+', size='+str(ps)+'x'+str(ps)+'x'+str(ps)
        fig = plt.figure(figsize=(len(patch)/2,4))
        plt.axis('off')
        plt.title(title)
        imr = self.rt(vt.norm2filledmgcc(patch.detach())).cpu()[:,0]
        im = self.upsample_im(patch.detach()[:,:,ps//2]).cpu()[:,0]
        for i in range(len(patch)):
            sub = fig.add_subplot(4,len(patch)//2,  i + 1)
            sub.axis('off')
            sub.imshow(im[i], vmin = 0, vmax = 1)
            sub = fig.add_subplot(4,len(patch)//2, len(patch) + i + 1)
            sub.axis('off')
            sub.imshow(imr[i])
        cfn = self.cfnf if kind == 'fake' else self.cfnr
        fn_path = cfn(fn)
        os.makedirs(os.path.dirname(fn_path), exist_ok=True)  # 🔧 CREA LA CARPETA SI NO EXISTE

        try:
            plt.savefig(fn_path, dpi=self.dpi)
        except Exception as e:
            print(f"[ERROR] Falló al guardar imagen: {fn_path} -- {e}")
        finally:
            plt.close(fig)

# save training errors
class ErrorFig:
    def __init__(self, p_out, epochs = 0):
        vals = p_out.split('\\')[-1]
        self.title = 'errors: '+vals
        if not os.path.exists(p_out): os.makedirs(p_out)
        self.cfn = os.path.join(p_out, 'error-values.svg')
        self.ptfn = os.path.join(p_out, 'error-values.pt')
        self.values = {}
        if os.path.exists(self.ptfn) and epochs > 0 :
            ef_values = torch.load(self.ptfn)
            for key in ef_values.keys():
                self.values[key] = ef_values[key][:epochs]
        
    def save(self):
        torch.save(self.values, self.ptfn)    
                
    def __call__(self, values):
        for key in values.keys():
            if key not in self.values.keys():
                self.values[key] = []
            self.values[key].append(values[key])
            xmax= len(self.values[key])
        epochs = torch.arange(xmax)  
            
        fig = plt.figure(figsize=(10,4))
        plt.title(self.title)
        for ikey, key in enumerate(self.values.keys()):
            plt.plot(epochs, self.values[key], label = key, color= 'C'+str(ikey))
            
        plt.xlabel('epoch')   
        plt.xlim(-0.1,xmax+0.1)
        plt.ylim(-0.1,4.1 )
        plt.grid()
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)
        plt.savefig(self.cfn)
        plt.close(fig)


def path_length_regularization(fake, latents, epsilon=1e-8):
    # fake: [B, C, D, H, W]
    # latents: [B, latent_dim]
    noise = torch.randn_like(fake) / (fake.numel() / fake.shape[0])**0.5 

    grad = torch.autograd.grad(
        outputs=(fake * noise).sum(),
        inputs=latents,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]  # grad shape: [B, latent_dim]

    path_lengths = torch.sqrt(grad.pow(2).sum(dim=1) + epsilon) 
    return (path_lengths - path_lengths.mean()).pow(2).mean()
