import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

import argparse, torch, numpy, tqdm, time, shutil
import utils, bonedataset, model 
from structuralparameters import BoneParameters as BoneParam

PATH_OUT = '../out'

parser = argparse.ArgumentParser()

# important parameters
parser.add_argument('--cont_train', type=int, default=-1, help='Continue with pre-trained network at epoch [N = Epoch N | 0 = start new | -1 = last checkpoint]')
parser.add_argument('--patch_size', type=int, default=32, help='Patch size (p x p x p)')
parser.add_argument('--nch_G', type=int, default=96, help='Number of hidden channels')
parser.add_argument('--depth_G', type=int, default=2, help='Depth of NNs')
parser.add_argument('--nch_D', type=int, default=16, help='Number of hidden channels')
parser.add_argument('--depth_D', type=int, default=3, help='Depth of NNs')
parser.add_argument('--batch_size', type=int, default=32, help='Size of the training batch per iteration')
parser.add_argument('--lr', type=float, default=2e-5, help='Initial learning rate')
##GRAFIENTE ORIGINAL 5
parser.add_argument('--lam_gradient', type=float, default=3, help='Lambda gradient for critic')
parser.add_argument('--lam_drift', type=float, default=0.5, help='Lambda range for critic')
#original 1 ssim
parser.add_argument('--lam_ssim', type=float, default=1.0, help='Lambda ssim penalty for generator')
parser.add_argument('--in_nch', type=int, default=32, help='Number of input channels')
parser.add_argument('--n_iter', type=int, default=15, help='Number of epochs to train before changing the progress')
parser.add_argument('--maxoptsteps', type=int, default=16, help='Maximum number of iterations of one NN before training the opponent')
parser.add_argument('--dataloader_length', type=int, default=2**11, help='Batches per epoch')
parser.add_argument('--bones_per_epoch', type=int, default=5, help='Bones per epoch')
parser.add_argument('--epochs', type=int, default=171, help='Number of epochs to train (multiple of model_checkpoints')
parser.add_argument('--footprint_SSIM', type=int, default=16, help='Footprint for computation of SSIM')
parser.add_argument('--n_seekreal', type=int, default=64, help='Number of patches for reference seeking')
parser.add_argument('--device', type=str, default='cuda', help='Training device. [cuda|cpu]. Default: cuda')
parser.add_argument('--fix_random_seed', type=int, default=12, help='Fix random seed')
parser.add_argument('--model_checkpoints', type=int, default=10, help='Checkpoints interval')
parser.add_argument('--screenshot_interval', type=int, default=30, help='min. screenshot interval in sec.')


###############################################################################
def _path_out(G,D):
    G_name = "Generator" if isinstance(G, torch.nn.DataParallel) else str(G)
    D_name = "Discriminator" if isinstance(D, torch.nn.DataParallel) else str(D)
    
    return os.path.join(PATH_OUT, G_name + "_" + D_name)

def load_NNs(opt):
    print("Create networks", end = '...')
    G = model.Generator(output_shape=opt.patch_size, size_in=opt.in_nch, nch=opt.nch_G)
    D = model.Discriminator(output_shape=opt.patch_size, nch=opt.nch_D)

    if torch.cuda.device_count() > 1:
        print(f"num {torch.cuda.device_count()} GPUs")
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)

    # Enviar los modelos a GPU
    G = G.to(opt.device)
    D = D.to(opt.device)
    p_out = _path_out(G,D) 
    path = os.path.join(p_out, 'models')
    print('done.')
    print('Generator has',sum(p.numel() for p in G.parameters()),'parameters')
    print('Discriminator has',sum(p.numel() for p in D.parameters()),'parameters')
    load_ = False
    if opt.cont_train != 0 and os.path.exists(path):
        fns = os.listdir(path)
        if len(fns)>0: 
            epochs= [int(ep.split('-')[-1].split('.')[0]) for ep in fns]
            if opt.cont_train == -1: 
                opt.cont_train = max(epochs)
            if opt.cont_train in epochs:
                load_ = True
                fn = '_epoch-{}.pt'.format(opt.cont_train)
                ###############################################################
                G_state_dict = torch.load(os.path.join(path, 'G'+fn), map_location=opt.device)
                D_state_dict = torch.load(os.path.join(path, 'D'+fn), map_location=opt.device)

                if isinstance(G, torch.nn.DataParallel):
                    G_state_dict = {"module." + k: v for k, v in G_state_dict.items()}

                if isinstance(D, torch.nn.DataParallel):
                    D_state_dict = {"module." + k: v for k, v in D_state_dict.items()}

                G.load_state_dict(G_state_dict, strict=False)
                D.load_state_dict(D_state_dict, strict=False)


                print('Continue training of existing GAN at epoch {}.'.format(opt.cont_train))
    opt.cont_train *= load_             
    if opt.cont_train == 0:
        if os.path.exists(p_out):
            try:
                shutil.rmtree(p_out)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))       
        os.makedirs(path)
        print('Start training of new GAN.') 
    return G, D
  
def saveNNs(G, D, epoch):
    fn = '_epoch-{}.pt'.format(epoch)
    p = os.path.join(_path_out(G, D) , 'models')
    
    torch.save(G.module.state_dict() if isinstance(G, torch.nn.DataParallel) else G.state_dict(), os.path.join(p, 'G'+fn))
    torch.save(D.module.state_dict() if isinstance(D, torch.nn.DataParallel) else D.state_dict(), os.path.join(p, 'D'+fn))

     
def plot_histograms(opt, boneparam, G, path_out, epoch, N=2**10):

    par_raw = torch.zeros((0, len(boneparam)), device=boneparam.device)
    G.eval()
    with torch.no_grad():
        for i in range(0, N, 16):
            z = torch.randn(opt.batch_size, G.module.size_in if isinstance(G, torch.nn.DataParallel) else G.size_in, device=boneparam.device)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
            z.requires_grad_(True) 
            fake = G(z)
            par_raw = torch.cat([par_raw, boneparam(fake, type_out=BoneParam.PAR_RAW)])
    
    par_norm = boneparam(par_raw, BoneParam.PAR_RAW, BoneParam.PAR_NORM)
    par_pca = boneparam(par_norm, BoneParam.PAR_NORM, BoneParam.PAR_PCA)

    try:
        sp_r = boneparam._get_real_parameters()
        sp_pca_real = boneparam(sp_r, BoneParam.PAR_RAW, BoneParam.PAR_PCA)
        pc1_real = sp_pca_real[:, 0]
        pc1_fake = par_pca[:, 0]
        varianza_real = pc1_real.var().item()
        varianza_fake = pc1_fake.var().item()
        varianza_explicada = varianza_fake / varianza_real
    except Exception as e:
        print(f"No se pudieron calcular los datos reales. Error: {e}")
        pc1_real, pc1_fake, varianza_explicada = None, None, None

    # Guardar histogramas
    p = os.path.join(path_out, 'histograms')
    if not os.path.exists(p): os.makedirs(p)
    boneparam.histogram_plot(par_raw, BoneParam.PAR_RAW, os.path.join(p, f'hist_raw_epoch-{epoch}.png'))
    boneparam.histogram_plot(par_norm, BoneParam.PAR_NORM, os.path.join(p, f'hist_norm_epoch-{epoch}.png'))
    boneparam.histogram_plot(par_pca, BoneParam.PAR_PCA, os.path.join(p, f'hist_pca_epoch-{epoch}.png'))

    return pc1_real, pc1_fake, varianza_explicada
    
def plot_reference_images(opt, sb, n_plot):
    fixed_real_patch = torch.zeros(n_plot, 1, opt.patch_size, opt.patch_size, opt.patch_size, device=opt.device)
    dataset = bonedataset.Dataset(opt.patch_size, length=1, bpe=1, load_first_batch=False)
    for i in tqdm.tqdm(range(n_plot), desc="Generate reference images"):
        dataset.next_epoch()
        fixed_real_patch[i] = dataset[0]
    sb(fixed_real_patch, epoch=0, kind='real')

def get_opt_boneparam():

    opt = parser.parse_args()
    boneparam = BoneParam(opt.patch_size, device=opt.device)
    return opt, boneparam

# train
def train(opt):
    torch.autograd.set_detect_anomaly(True)

    print("----- TRAIN: PARAMETERS -----")
    print(opt)
    print("-----------------------------")
    
    torch.manual_seed(opt.fix_random_seed)
    opt.device = opt.device if torch.cuda.is_available() else 'cpu'
    print("run training on " + opt.device)
    
    # Load NNs
    G, D = load_NNs(opt)
    path_out = _path_out(G, D)
    print('train net ' + path_out)
    
    # Load Dataset
    print("Load dataset", end = '...')
    dataset_length = opt.dataloader_length * opt.batch_size
    dataset = bonedataset.Dataset(opt.patch_size, length = dataset_length, 
                                  bpe = opt.bones_per_epoch, load_first_batch=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)
    print('done.')
    
    boneparam = BoneParam(opt.patch_size, device = opt.device)
    ssim_pen = utils.SSIMPenalty(footprint = opt.footprint_SSIM)
    grad_pen = utils.GradientPenalty(D.module if isinstance(D, torch.nn.DataParallel) else D)
    sb = utils.SaveBatch(path_out, device = opt.device)
    ef = utils.ErrorFig(path_out, opt.cont_train)
    z_fixed = torch.randn(16, opt.in_nch, device=opt.device) 
    z_fixed = utils.hypersphere(z_fixed)  
    plot_reference_images(opt, sb, z_fixed.shape[0])

    print("Run training loop...")
    optimizerG = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(0, 0.999))
    optimizerD = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(0, 0.999))
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=opt.epochs, eta_min=1e-6)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=opt.epochs, eta_min=1e-6)
    loss_items = ['l_crit', 'l_gen', 'l_drift', 'l_grad', 'l_ssim', 'l_r1', 'l_pathreg']

    tanh1 = torch.tanh(torch.tensor(1)).item() 
    time_0 = 0 
    i_opt = 1.0 
    epoch0 = (opt.cont_train + 1) * (opt.cont_train > 0) 
    _lastv = lambda key: '-' if len(loss_epoch[key])==0 else "{:.3f}".format(numpy.mean(loss_epoch[key])) 
        
    with tqdm.tqdm(total = len(dataloader)) as pbar:
        for epoch in range(epoch0, epoch0+opt.epochs):
            pbar.reset()
            dataset.next_epoch()
            loss_epoch = {}
            for item in loss_items: loss_epoch[item] = []
            ssimpen_real = 0
            
            for i, real in enumerate(dataloader):

                if i == 0:
                    print(f"[epoch {epoch} | iter {i}]")
                str_ = "[{:3d}] D/G={:3d}/{:3d}  crit={}  gen={}  grad={}  drift={}  ssim={}  r1={}  pathreg={}".format(
                    epoch, len(loss_epoch['l_crit']), len(loss_epoch['l_gen']),
                    _lastv('l_crit'), _lastv('l_gen'), _lastv('l_grad'),
                    _lastv('l_drift'), _lastv('l_ssim'), _lastv('l_r1'), _lastv('l_pathreg'))


                pbar.set_description(str_)
                pbar.update()
                real = real.to(opt.device)

                D_real_output = D.module(real) if isinstance(D, torch.nn.DataParallel) else D(real)
                D_real_tm = D_real_output.mean().unsqueeze(0)


                if i == 0:
                    with torch.no_grad():
                        ssimpen_real = ssim_pen(real).max().item()

                if i_opt // 1: # i_opt between 1 and 2
                # ============= Train the discriminator =============#
                # l_gan: achieve higher values for real than for fake patches
                # l_drift: force to have values of 1 and -1 for real and fake
                # l_grad: force internal smoothness of the discriminator
                    G.eval()
                    D.train()
                    D.zero_grad()
                    with torch.no_grad():     
                        z = torch.randn(opt.batch_size, G.module.size_in if isinstance(G, torch.nn.DataParallel) else G.size_in, device=boneparam.device)
                        z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
                        fake = G.module(z) if isinstance(G, torch.nn.DataParallel) else G(z)
 
                    D_fake_output = D.module(fake) if isinstance(D, torch.nn.DataParallel) else D(fake)
                    D_fake_tm = D_fake_output.mean().unsqueeze(0)

                    #R1 REGULARIZATION
                    real.requires_grad_(True)
                    D_real = D.module(real) if isinstance(D, torch.nn.DataParallel) else D(real)
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=real,
                        create_graph=True, retain_graph=True, only_inputs=True
                    )[0]
                    r1_penalty = grad_real.pow(2).view(real.size(0), -1).sum(1).mean()
                    l_r1 = 10.0 * r1_penalty  # λ = 10.0
                    loss_epoch.setdefault('l_r1', []).append(l_r1.item())
                    # gradient penalty loss
                    l_grad = (grad_pen(real.data, fake.data)* opt.lam_gradient)**0.5      
                    # drift loss
                    l_drift = ((D_fake_tm+tanh1)**2 + (D_real_tm-tanh1)**2)*opt.lam_drift 
                    # direct critic loss
                    l_gan = torch.nn.ReLU()(tanh1*2 + D_fake_tm - D_real_tm)/(2*tanh1)   
                    # combined critic loss
                    l_ccomb = l_gan + l_grad + l_drift + l_r1
                    # update
                    torch.autograd.set_detect_anomaly(True)

                    l_ccomb.backward()
                    torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0) 
                    optimizerD.step()
                    loss_epoch['l_crit'].append(l_gan.item())
                    loss_epoch['l_grad'].append(l_grad.item())
                    loss_epoch['l_drift'].append(l_drift.item())
                    
                else:  # i_opt between 0 and 1   
                    # ============= Train the generator =============#
                    # l_gan: achieve lower values for real than for fake patches
                    # l_ssim: force to have patches of different appearances when z is different
                    D.eval()
                    G.train()
                    G.zero_grad()
                    z = torch.randn(opt.batch_size, G.module.size_in if isinstance(G, torch.nn.DataParallel) else G.size_in, device=boneparam.device)
                    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)

                    fake = G.module(z) if isinstance(G, torch.nn.DataParallel) else G(z)
                    D_fake_output = D.module(fake) if isinstance(D, torch.nn.DataParallel) else D(fake)
                    D_fake_tm = torch.tanh(D_fake_output).mean().unsqueeze(0)  
                    # structural similarity loss
                    lam_ssim_value = opt.lam_ssim * (1 - epoch / opt.epochs)

                    if epoch >= 1:
                        penalty = ssim_pen(fake, z) - ssimpen_real
                        if torch.isnan(penalty).any() or torch.isinf(penalty).any():
                            l_ssim = torch.tensor(0.0, device=opt.device)
                        else:
                            l_ssim = utils.fuzzymax(torch.nn.ReLU()(torch.tanh(penalty))).clamp(min=0, max=1e6) * lam_ssim_value


                    else:
                        l_ssim = torch.tensor(0.0, device=opt.device)
                    
                    l_pathreg = torch.tensor(0.0, device=opt.device)
                    if epoch >= 5 and (i % 16 == 0):
                        z_path = torch.randn(opt.batch_size, G.module.size_in if isinstance(G, torch.nn.DataParallel) else G.size_in, device=boneparam.device)
                        z_path = z_path / (z_path.norm(dim=1, keepdim=True) + 1e-8)
                        z_path.requires_grad_(True)
                        fake_path = G.module(z_path) if isinstance(G, torch.nn.DataParallel) else G(z_path)
                        lambda_path = min(1.0, (epoch - 5) / 25.0) * 0.5
                        l_pathreg = utils.path_length_regularization(fake_path, z_path) * lambda_path
                        loss_epoch.setdefault('l_pathreg', []).append(l_pathreg.item())
                    else:
                        loss_epoch.setdefault('l_pathreg', []).append(0.0) 

                    l_gan = torch.nn.ReLU()(tanh1*2 + D_real_tm - D_fake_tm)/(2*tanh1)
                    l_gcomb = l_gan + l_ssim + l_pathreg

                    # update
                    l_gcomb.backward()
                    torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0) 
                    optimizerG.step()
                    loss_epoch['l_gen'].append(l_gan.item())
                    loss_epoch['l_ssim'].append(l_ssim.item())
            
                i_opt = (i_opt + 1/opt.maxoptsteps) % 2 if l_gan.item()>1 else (i_opt//1+1)%2
                
            # ============= show training status =============# 
            if not epoch % opt.model_checkpoints and epoch > 0:
                saveNNs(G, D, epoch)
                ef.save()
    
            pc1_real, pc1_fake, varianza_explicada = plot_histograms(
                opt,
                boneparam, 
                G.module if isinstance(G, torch.nn.DataParallel) else G, 
                path_out, 
                epoch
            )

            if varianza_explicada is not None:
                print(f"variance explained (epoc {epoch}): {varianza_explicada}")
                with open(os.path.join(path_out, 'variance explained.log'), 'a') as log_file:
                    log_file.write(f"Epoch {epoch}: {varianza_explicada}\n")
            else:
                print(f"The variance explained at the time could not be calculated {epoch}.")
            
            val = {}
            for item in loss_items: val[item] = round(numpy.mean(loss_epoch[item]),3)
            ngen = len(loss_epoch['l_gen'])
            ncrit = len(loss_epoch['l_crit'])
            val['D/(G+D)'] = round(ncrit / (ngen + ncrit), 3)
            ef(val)
            time_1 = time.time()
            if (time_1-time_0)> opt.screenshot_interval:
                fixed_fake_patch = G.module(z_fixed) if isinstance(G, torch.nn.DataParallel) else G(z_fixed)

                sb(fixed_fake_patch, epoch)
                time_0 = time_1
            schedulerG.step()
            schedulerD.step()

    print("Training finished.")
    return G  
        
if __name__ == "__main__":
    opt, boneparam = get_opt_boneparam()
    train(opt) 