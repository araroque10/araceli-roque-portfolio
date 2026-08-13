import torch, tqdm, os, sklearn.decomposition, seaborn, scipy.special, matplotlib
import bonedataset
import matplotlib.pyplot as plt
matplotlib.use('Agg')

class PowerTransformer:
    def __init__(self, parameters = None):
        self.parameters = parameters
    
    # X: array-like of shape (n_samples, n_features)
    # requires parameters not empty
    def __call__(self, X, inverse = False):
        p = self.parameters
        assert not p is None, 'PowerTransform not initialized yet'
        if inverse:    
            Y = (PowerTransformer._yj((X - p[4]) / p[3], p[0], inverse) - p[2]) / p[1]
        else:
            Y = PowerTransformer._yj(X * p[1] + p[2], p[0], inverse) * p[3] + p[4]
        return Y
          
    def _yj(data, _lambda, inverse = False):
        s = (data>=0) * 2 - 1    
        lam = 1 - s * (1 - _lambda)
        use_log = (lam.abs()==0) * 1
        if inverse:
            ret = s * (torch.exp(s*use_log*data)-1 + (1-use_log) * ((s*data*lam+1)**(1/(lam+use_log))-1))
        else:
            ret = s * (torch.log(s*use_log*data + 1) + (1-use_log)* ((s*data + 1)**lam-1)/(lam+use_log)) 
        return ret
    
    def fit(self, X, max_it = 5e4, max_step = 2e3, N = 101, crit = torch.nn.MSELoss(), 
                 optimizer = torch.optim.Adamax, lr = 0.002, weight = lambda q: torch.abs(q)+1, 
                 max_lambda = 3.0):
        self.parameters = torch.zeros((5,X.shape[1]))
        for idx in range(X.shape[1]):
            torch.cuda.empty_cache()
            str_in = "[{}] - ".format(idx+1)
            fitted = self._fit_one(X[:,idx].detach().clone(), max_it, max_step, N, crit, optimizer, lr, weight, max_lambda, str_in)
            self.parameters[:,idx] = torch.tensor(fitted)
    
    def _norm(quant):
        s = 2/(quant[-1]-quant[0])
        m = -(1 + 2*quant[0] / (quant[-1]-quant[0]))
        return quant * s + m, s, m 
    
    # fit one by one. I didn't figure out yet how to efficiently optimize all at once...    
    def _fit_one(self, X, max_it, max_step, N, crit, optimizer, lr, weight, max_lambda, str_in =''):
        q = torch.linspace(0, 1, N, device = X.device)[1:-1]
        quant_gauss, sg, _ = PowerTransformer._norm(2**0.5 * scipy.special.erfinv(2*q.cpu()-1).to(X.device))
        w = weight(quant_gauss)
        quant_in = torch.quantile(X, q).detach().clone()
        quant_n, s_in, m_in = PowerTransformer._norm(quant_in)
        par_opt = torch.tensor([0.0, 0.0, 1.0], device = X.device).requires_grad_()
        optimizer = optimizer([par_opt], lr = lr)
        best_loss = torch.inf
        best_par = par_opt.detach().clone()
        i = 0
        last_i = 0
        with tqdm.tqdm() as pbar:
            while i<max_it and (i-last_i)<max_step:
                i+=1
                if not i%500:
                    str_ =  str_in +"loss:{} - lambda:{}".format(round(best_loss.item(),6), round(best_par[2].item(),3)) 
                    pbar.set_description(str_)
                    pbar.update(500)
                optimizer.zero_grad()
                q_in = (quant_n + par_opt[0]) * 2**par_opt[1]
                q_gauss = PowerTransformer._yj(q_in, par_opt[2])
                q_back = PowerTransformer._yj(q_gauss, par_opt[2], inverse = True)
                data_opt,_,_ = PowerTransformer._norm(q_gauss)
                loss_trans = crit(q_in * w, q_back * w)*5
                loss_map = crit(data_opt * w, quant_gauss * w)
                loss_biglambda = torch.nn.ReLU()(torch.abs(par_opt[2]-1)-max_lambda)
                loss = loss_map + loss_biglambda + loss_trans
                if loss < best_loss:
                    best_par = par_opt.clone().detach()
                    best_loss = loss.clone().detach()
                    last_i = i    
                loss.backward()
                optimizer.step()
        # improve stability: if lambda close to 0 or 2 go directly to log-term        
        hs= torch.nn.Hardshrink(0.01)
        lam_opt = hs(hs(best_par[2])-2).item()+2
        # get compound pre-fit slope and intercept
        s_opt = (2**best_par[1] * s_in).item()
        m_opt = ((m_in + best_par[0]) * 2**best_par[1]).item() 
        # get post-fit slope and intercept
        _, s_out, m_out, = PowerTransformer._norm(PowerTransformer._yj(quant_in * s_opt + m_opt, lam_opt))
        return lam_opt, s_opt, m_opt, s_out/sg, m_out/sg

class BoneParameters:
    PATH_TRANS = "/usuarios/araceli.roque/evolutivo/data/transforms"
    PATH_PLOTS = '../out/real_data_transforms'
    #PATH_TRANS = "./data/transforms"
    #PATH_PLOTS = '../out/real_data_transforms'

    VOL_NORM, VOL_HU, VOL_DENS, PAR_RAW, PAR_NORM, PAR_PCA = -2, -1, 0, 1, 2, 3
    
    def __init__(self, patch_size = 32, spacing = 0.164, t = 200., sigma = 45.,  
                 use_transforms = True, recompute = False, device = 'cpu'):
        symb = lambda i: str(round(i))
        self._str = 'BP-{}-{}-{}-{}'.format(symb(patch_size),symb(t), symb(sigma), symb(spacing*1000))
        self.kt = .945 / 4.0 * torch.tensor([[[[[1.,1.],[1.,1.]],[[-1.,-1.],[-1.,-1.]]]],
                           [[[[1.,1.],[-1.,-1.]],[[1.,1.],[-1.,-1.]]]],
                           [[[[1.,-1.],[1.,-1.]],[[1.,-1.],[1.,-1.]]]]]).to(device) 
        self.patch_size = patch_size
        self.t = t
        self.sigma = sigma    
        self.spacing = spacing
        self.device = device
        self.use_transforms = use_transforms
        self.downsample = torch.nn.AvgPool3d(kernel_size = 2, stride = 2)
        if use_transforms:    
            if not os.path.exists(BoneParameters.PATH_TRANS): os.makedirs(BoneParameters.PATH_TRANS)
            if not os.path.exists(os.path.join(BoneParameters.PATH_TRANS, str(self)+'.pt')) or recompute: 
                print('generate normalization file for setting ' + str(self) +':')
                self._fit()             
            else:
                print('load normalization file of setting ' + str(self))
                self._load()
    
    def __str__(self):
        return self._str
    
    # map data from type_in to type_out 
    # backwards: only PAR_PCA down to PAR_RAW
    # type VOL_GAN: volume normalized for GAN
    # type VOL_HU: volume in HU with air
    # type VOL_MGCC: volume in mg/cc with filled air
    # type PAR_RAW: structural parameters
    # type PAR_NORM: normalized structural parameters
    # type PAR_PCA: pca transformed set of parameters
    # default: from GAN to PCA (-2 to 3)
    def __call__(self, data, type_in = VOL_NORM, type_out = PAR_PCA):
        f = [bonedataset.VoxelTransform.denormalize_hu,
             bonedataset.VoxelTransform.hu2filledmgcc,
             self._comp_parameters, self._normalize, self._princomp]
        if type_in<type_out: 
            for idx in range(type_in, type_out, 1): data = f[idx+2](data)
        if type_in>type_out: 
            for idx in range(type_in, type_out, -1): data = f[idx+1](data, inverse = True)
        return data
    
    def _tmd(self, data, pbv):
        alpha_tmd = 1.4
        eps_bmd = 5
        eps_bvtv = 0.015
        tmd_tv = torch.mean((torch.nn.functional.softplus(data, 1/eps_bmd) * pbv).view((*data.size()[:2], -1)), dim = -1)
        weights = torch.nn.functional.softplus(torch.mean((pbv**alpha_tmd).view((*data.size()[:2], -1)), dim = -1), 1/eps_bvtv)
        tmd = tmd_tv / weights
        return tmd
     
    def _bstv(self, pbv):
        grad_sqr = torch.pow(torch.nn.functional.conv3d(pbv, self.kt), 2)
        mag = torch.pow(torch.sum(grad_sqr, dim = 1, keepdims = True), 0.5) / self.spacing
        bstv = torch.mean(mag.view((*mag.size()[:2], -1)), dim = -1)
        return bstv
    
    def _comp_parameters(self, data):
        assert self.patch_size == data.shape[2], 'patch_size must be '+ str(self.patch_size)
        eps_bstv = 0.1
        bmd = data.view(*data.shape[:-3], -1).mean(dim = -1)
        bmd_sd = data.view(*data.size()[:-3], -1).std(dim = -1)
        pbv_ = torch.sigmoid((data - self.t) / self.sigma) 
        tmd = self._tmd(data, pbv_)
        bvtv = torch.mean(pbv_.view((*pbv_.size()[:2], -1)), dim = -1) # = 1-mvtv
        bstv = self._bstv(pbv_)
        bstv_plus_ = torch.nn.functional.softplus(bstv, 1/eps_bstv)
        tbth = 2 * bvtv / bstv_plus_ # = 1/tb_n
        tbsp = 2 * (1 - bvtv) / bstv_plus_    
        return torch.cat([bmd, bmd_sd, tmd, bvtv, bstv, tbth, tbsp],dim = 1)

    hist_limits =[(0,600), (0,400), (200,600), (0,1), (0,2.5), (0,1.0), (0,4)]
    
    def get_names(type_ = 1):
        names = ['BMD', 'BMD.SD', 'TMD', 'BV/TV', 'BS/TV', 'Tb.Th', 'Tb.Sp']
        if type_ == 2:
            names = ['norm.'+par for par in names]
        if type_== 3:
            names = ['PC-'+str(idx+1) for idx,_ in enumerate(names)]    
        return names
  
    def __len__(self): return len(BoneParameters.get_names())
    
    # inverse: denormalize normalized parameters  
    def _normalize(self, data, inverse = False):
        return self.PT(data, inverse = inverse)
    
    # inverse: transform from pca to normalized parameters    
    def _princomp(self, data, inverse = False):
          mat = self.mat_inv_pca if inverse else self.mat_pca  
          return data @ mat
        
    #---------- private functions --------------# 
    # compute what is group 1, 2 and 3 at sp_wang (data is in mg/cc)
    def _load(self):
        file_ = torch.load(os.path.join(BoneParameters.PATH_TRANS, str(self)+'.pt'))
        self.PT = PowerTransformer(file_['PT.parameters'].to(self.device))
        self.explained_variance = file_['explained_variance']
        self.mat_pca = file_['mat_pca'].to(self.device)
        self.mat_inv_pca = file_['mat_inv_pca'].to(self.device)
        if 'hist' in file_:
            self.hist = file_['hist']
            self.bins = file_['bins']
    
    # get type 1 parameters of real dataset
    def _get_real_parameters(self, lengthpb = 2**6):
        dataset = bonedataset.Dataset(self.patch_size, length = lengthpb*5, bpe = 5, load_first_batch=False, device = self.device)
        dataloader = torch.utils.data.DataLoader(dataset, 32, shuffle=False, drop_last=False)
        struct_param = torch.zeros(0, len(self.get_names()), device = self.device)
        with tqdm.tqdm() as pbar:
            for epoch in range(dataset.epochs()):
                dataset.next_epoch()
                for real in dataloader:
                    pbar.update(real.shape[0])
                    c_par = self(real, type_out = BoneParameters.PAR_RAW)
                    struct_param = torch.cat([struct_param, c_par])
        return struct_param           
    
    def print_stats(sp_raw, sp_n):
        q = torch.tensor([0.01,0.25,0.5,0.75,0.99], device = sp_raw.device)
        quantr = torch.quantile(sp_raw, q, dim =0)
        quantn = torch.quantile(sp_n, q, dim =0)
        namesr = BoneParameters.get_names(BoneParameters.PAR_RAW)
        namesn = BoneParameters.get_names(BoneParameters.PAR_NORM)
        print("---stats of raw parameters----------")
        print("rmin - Q1 - Q2 - Q3 - rmax")
        for idx in range(quantr.shape[1]):
            print('{:<12}: {:.2f} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(
                namesr[idx], *quantr[:,idx]))
        print("---stats of normalized parameters---")
        for idx in range(quantr.shape[1]):
            print('{:<12}: {:.2f} - {:.2f} - {:.2f} - {:.2f} - {:.2f}'.format(
                namesn[idx], *quantn[:,idx]))
        print("-----------")
        
    # fitting may be faster in cpu (it is what we have so far)
    def _fit(self, lengthpb = 2**10, fit_device = 'cpu'):
        dict_ = {}
        print('Get real structural parameters...')
        sp_r = self._get_real_parameters(lengthpb)
        print('Fit real distributions to Gaussians...')
        PT = PowerTransformer()
        PT.fit(sp_r.to(fit_device))
        PT.parameters = PT.parameters.to(sp_r.device)
        dict_['PT.parameters'] = PT.parameters.cpu()
        norm_struct_param = PT(sp_r).detach().cpu()
        pca = sklearn.decomposition.PCA()
        pca.fit(norm_struct_param.numpy())
        dict_['explained_variance'] = torch.tensor(pca.explained_variance_ratio_)
        _evals_sqrt = torch.tensor(pca.explained_variance_**0.5)
        _evecs = torch.tensor(pca.components_.T)
        dict_['mat_pca'] = (_evecs / _evals_sqrt.unsqueeze(0)).to(norm_struct_param.dtype)
        dict_['mat_inv_pca'] = (_evecs.T * _evals_sqrt.unsqueeze(1)).to(norm_struct_param.dtype)
        torch.save(dict_, os.path.join(BoneParameters.PATH_TRANS,  str(self)+'.pt'))
        
        self._load()
        sp_n = self(sp_r, BoneParameters.PAR_RAW, BoneParameters.PAR_NORM)
        sp_pc = self(sp_r, BoneParameters.PAR_RAW, BoneParameters.PAR_PCA)
        
        BoneParameters.print_stats(sp_r, sp_n)
        print('Make quality assurance plots...')
        _hist = torch.zeros(3,100,sp_r.shape[1])
        _bins = torch.zeros(3,101,sp_r.shape[1])
        bins_g = torch.linspace(-4,4,101)
        for idx in range(sp_r.shape[1]):
            bins_r = torch.linspace(*BoneParameters.hist_limits[idx],101)
            _hist[0,:,idx], _bins[0,:,idx] = torch.histogram(sp_r.cpu()[:,idx], bins = bins_r)
            _hist[1,:,idx], _bins[1,:,idx] = torch.histogram(sp_n.cpu()[:,idx], bins = bins_g)
            _hist[2,:,idx], _bins[2,:,idx] = torch.histogram(sp_pc.cpu()[:,idx], bins = bins_g)
        dict_['hist'] = _hist
        dict_['bins'] = _bins        
        torch.save(dict_, os.path.join(BoneParameters.PATH_TRANS,  str(self)+'.pt'))    
        
        self._load()
        # plots:
        delta_ba = (len(sp_r) // 5000) + 1 
        folder = os.path.join(BoneParameters.PATH_PLOTS, str(self))
        if not os.path.exists(folder): os.makedirs(folder)
        self.histogram_plot(None, BoneParameters.PAR_RAW, os.path.join(folder, 'hist_raw.png'))
        self.histogram_plot(None, BoneParameters.PAR_NORM, os.path.join(folder, 'hist_norm.png'))
        self.histogram_plot(None, BoneParameters.PAR_PCA, os.path.join(folder, 'hist_pca.png'))
       
        sp_rback = self(sp_pc, BoneParameters.PAR_PCA, BoneParameters.PAR_RAW) 
        sp_nback = self(sp_pc, BoneParameters.PAR_PCA, BoneParameters.PAR_NORM) 
        BoneParameters.blend_altman_plot(sp_r.cpu()[::delta_ba], sp_rback.cpu()[::delta_ba], 
                          os.path.join(folder, 'blend-altman_powertransform_raw.png'), BoneParameters.PAR_RAW)
        BoneParameters.blend_altman_plot(sp_n.cpu()[::delta_ba], sp_nback.cpu()[::delta_ba], 
                          os.path.join(folder, 'blend-altman_powertransform_norm.png'), BoneParameters.PAR_NORM)
        BoneParameters.hist2d_plot(sp_pc.cpu()[::delta_ba], 'PC 1 vs 2', os.path.join(folder,'PCA_1_vs_2.png'), 0, 1)
        BoneParameters.hist2d_plot(sp_pc.cpu()[::delta_ba], 'PC 1 vs 3', os.path.join(folder,'PCA_1_vs_3.png'), 0, 2)
        BoneParameters.hist2d_plot(sp_pc.cpu()[::delta_ba], 'PC 2 vs 3', os.path.join(folder,'PCA_2_vs_3.png'), 1, 2)
        print('done.')
        
    def histogram_plot(self, par, type_, fn):
        fig = plt.figure(figsize=(20,10))
        plt.clf()
        names = BoneParameters.get_names(type_)
        gauss = type_>1
        _h_real = self.hist[type_-1]
        _b_real = self.bins[type_-1]
        _bins_real = (_b_real[1:] + _b_real[:-1])/2
        _ch_real = torch.cumsum(_h_real, dim=0) 
        for idx in range(_h_real.shape[1]):
            b_real = _bins_real[:,idx]
            h_real = _h_real[:,idx]
            ch_real = _ch_real[:,idx]
            sub = fig.add_subplot(2, (_h_real.shape[1]+1)//2,  idx + 1)
            sub.set_title(names[idx])
            lim = [-4,4] if gauss else BoneParameters.hist_limits[idx]
            sub.set_xlim(*lim)
            sub.plot(b_real, h_real / h_real.max(),'C1', linewidth = 2, linestyle ='dashed', label = 'histogram of real data')
            sub.plot(b_real, ch_real / ch_real.max(),'C1', linewidth = 1,label = 'cumulative histogram of real data')
            if par is not None:
                h_fake, _b_fake = torch.histogram(par.cpu()[:,idx], bins = torch.linspace(*lim, 101))
                b_fake = (_b_fake[1:] + _b_fake[:-1])/2
                ch_fake = torch.cumsum(h_fake, dim=0)
                sub.plot(b_fake, h_fake / h_fake.max(),'C2', linewidth = 2, linestyle ='dashed', label = 'histogram of fake data')
                sub.plot(b_fake, ch_fake / ch_fake.max(),'C2', linewidth = 1,label = 'cumulative histogram of fake data')
            if idx == _h_real.shape[1]-1: plt.legend(loc='center',bbox_to_anchor=(1.25, 0.5, 0.5, 0.5))
        plt.savefig(fn, dpi = 300)
        plt.close(fig) 
        
    def blend_altman_plot(par1, par2, fn, type_=1):
        fig = plt.figure(figsize=(20,10))
        plt.clf()
        names = BoneParameters.get_names(type_)
        x = (par1 + par2)/2
        y = (par1 - par2)
        for i in range(par1.shape[1]):
            sub = fig.add_subplot(2, (par1.shape[1]+1)//2,  i + 1)
            sub.set_title(names[i])
            sub.plot(x[:,i],y[:,i] / x[:,i].std(),'.')
            plt.ylim(-0.0005,0.0005)
        plt.savefig(fn, dpi = 300)
        plt.close(fig)  
        
    def hist2d_plot(par, title, fn, x=0, y=1):
        names = BoneParameters.get_names(3)  
        fig = plt.figure(figsize = (10,10))
        cmap = 'coolwarm'
        plt.title(title)
        ax = seaborn.kdeplot(x = par.cpu()[:,x], y = par.cpu()[:,y], bw_adjust = 0.8,cmap=cmap, fill = True,levels = 30, thresh = 0.0)
        ax.set_facecolor(seaborn.color_palette(cmap, as_cmap = True)(255//60))
        seaborn.kdeplot(x = par.cpu()[:,x], y = par.cpu()[:,y], bw_adjust = 0.8,color = 'gray', levels = 10)
        plt.xlabel(names[x])
        plt.ylabel(names[y])
        plt.axis([-3, 3, -3, 3])
        plt.savefig(fn, dpi = 300)
        plt.close(fig) 
     
#%% check
if __name__ == '__main__':
    use_cuda = True
    device = 'cpu' if torch.cuda.is_available() and use_cuda else 'cpu'
    bp = BoneParameters(16, device = device)
    bp = BoneParameters(24, device = device)
    bp = BoneParameters(32, device = device, recompute=True)
    
   # bp.histogram_plot(par, 1, 'test.png')
    
#%%
    #new_data = torch.randn(16,len(bp))
    #cum_data = torch.zeros(0,7)
        
def _get_real_parameters(self, lengthpb = 2**6):
    print("Iniciando carga de datos...")
    dataset = bonedataset.Dataset(self.patch_size, length=lengthpb*5, bpe=5, load_first_batch=False, device=self.device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    struct_param = torch.zeros(0, len(self.get_names()), device=self.device)
    with tqdm.tqdm() as pbar:
        for epoch in range(dataset.epochs()):
            print(f"Procesando época {epoch + 1}/{dataset.epochs()}")
            dataset.next_epoch()  # Aquí podrías tener el problema de memoria
            for real in dataloader:
                pbar.update(real.shape[0])
                try:
                    c_par = self(real, type_out=BoneParameters.PAR_RAW)  # Procesa los datos
                except Exception as e:
                    print(f"Error durante el procesamiento de datos: {e}")
                    continue
                struct_param = torch.cat([struct_param, c_par])
    print("Finalizó la carga de datos.")
    return struct_param
