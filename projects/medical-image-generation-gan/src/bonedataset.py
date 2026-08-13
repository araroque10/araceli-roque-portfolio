

import os, torch, copy, tqdm
import numpy as np

print("Ejecutando bonedataset.py...")

class VoxelTransform:
    HU_MIN = -1400.
    HU_MAX = 4000.
    NORM_M = 300. # HU 300 -> 0.5 spongiosa
    NORM_S = 400. # HU 1500 -> 0.95 (cortical bone) HU -1000 -> 0.04 (water) 
    
    # to HU to GAN values
    def normalize_hu(data, m = NORM_M, s = NORM_S):
        return torch.sigmoid((data - m) / s)
      
    # to GAN to HU values
    def denormalize_hu(data, m = NORM_M, s = NORM_S):
        min_ = VoxelTransform.normalize_hu(torch.tensor(VoxelTransform.HU_MIN, device = data.device), m, s)
        max_ = VoxelTransform.normalize_hu(torch.tensor(VoxelTransform.HU_MAX, device = data.device), m, s)
        return -torch.log(1 / torch.clip(data, min_, max_) - 1) * s + m
    
    def hu2mgcc(data, inverse = False, mu_water = 0.25480, dens_slope =1.31704102e+03, dens_intercept = -3.36571991e+02):
        return (data / 1000 + 1) * mu_water * dens_slope + dens_intercept

    # HU with air to HU with water
    def fill_marrow(data, s = 100, x = [-1000, -560, 460], y= [-44, 284, 1010]):
        sp = torch.nn.Softplus()
        l1_ = sp((data-x[1])/s)*s* (y[2]-y[1]) / (x[2]-x[1])
        l2_ = sp(-(data-x[1])/s)*s* (y[1]-y[0]) / (x[1]-x[0])
        return l1_ - l2_ + y[1]

    def hu2filledmgcc(data):
        return VoxelTransform.hu2mgcc(VoxelTransform.fill_marrow(data))
    
    def norm2filledmgcc(data):
        return VoxelTransform.hu2filledmgcc(VoxelTransform.denormalize_hu(data))

class Dataset(torch.utils.data.Dataset):

    PATH = "/usuarios/araceli.roque/evolutivo/data/volumes"
    FN_INDEX = 'index.npz'

    PERMS = [lambda x: x, lambda x: x.flip(3), 
             lambda x: x.flip(3).flip(1), lambda x: x.flip(1),  
             lambda x: x.flip(3).transpose(2,3), lambda x: x.transpose(2,3), 
             lambda x: x.flip(1).transpose(2,3), lambda x: x.flip(3).flip(1).transpose(2,3), 
             lambda x: x.flip(3).flip(2), lambda x: x.flip(2), 
             lambda x: x.flip(2).flip(1), lambda x: x.flip(3).flip(2).flip(1),
             lambda x: x.flip(2).transpose(2,3), lambda x: x.flip(3).flip(2).transpose(2,3),
             lambda x: x.flip(3).flip(2).flip(1).transpose(2,3), lambda x: x.flip(2).flip(1).transpose(2,3)]
    
    def __init__(self, patchsize, length = 2**12, bpe = 8, permutate = True, fixed = False, 
                 load_first_batch = True, mapping = VoxelTransform.normalize_hu, device = 'cpu'):
        self.mapping = mapping
        self.vols = []
        self.patchsize = patchsize
        self.length = length
        self.permutate = permutate
        self.device = device 
        if fixed:
            self.random_choice = lambda a, s: np.linspace(0, a-1, s, dtype = 'int64')
        else:
            self.random_choice = lambda a, s: np.random.choice(a, s, replace = False)
        if not os.path.exists(os.path.join(Dataset.PATH, Dataset.FN_INDEX)): Dataset._make_index()
        file_index = np.load(os.path.join(Dataset.PATH, Dataset.FN_INDEX))
        numbers = file_index['numbers'][patchsize]
        select = numbers > (length // bpe + 1)
        self.fns = file_index['fns'][select]
        self.numbers = numbers[select]
        self.bpe = min(bpe, len(self.fns))
        self.length = min(min(self.numbers) * self.bpe, self.length)
        if self.length == 0: print('Warning: Dataset is empty!')
        self.idx_fn = len(self.fns)
        if load_first_batch: self.next_epoch()
       
    def __getitem__(self, idx):
        id_perm, idx_vol, x, y, z = self.indices[idx % len(self.indices)]
        patch = self.vols[idx_vol][:, x:x+self.patchsize,y:y+self.patchsize,z:z+self.patchsize]
        ret = Dataset.PERMS[id_perm](patch.clone())
        return ret
    
    def __len__(self): return self.length   
        
    def epochs(self): return len(self.fns) // self.bpe 
    
    def next_epoch(self):
        for vol in self.vols:
            del vol    
        torch.cuda.empty_cache()
        self.vols = []
        self.idx_fn += self.bpe
        if (self.idx_fn + self.bpe) > len(self.fns):
            rc = self.random_choice(len(self.fns),len(self.fns))
            self.fns = self.fns[rc]
            self.numbers = self.numbers[rc]
            self.idx_fn = 0
        fns = self.fns[self.idx_fn: self.idx_fn + self.bpe]
        numbers = self.numbers[self.idx_fn : self.idx_fn + self.bpe]
        numbers = np.maximum(1,np.round(numbers / np.sum(numbers) * self.length).astype(int))
        numbers[-1] = np.maximum(1,self.length - np.sum(numbers[:-1]))
        while np.sum(numbers)>self.length:
            numbers[np.argmax(numbers)] -= 1
        idx = np.zeros((3,0))
        id0 = np.zeros(0)
        for i, fn in enumerate(fns):
            npz = np.load(os.path.join(Dataset.PATH, fn))
            self.vols.append(self.mapping(torch.tensor(npz['slices'], 
                dtype=torch.float32, device = self.device).unsqueeze(0)))
            idx1 = np.array(np.where(npz['mask']>=self.patchsize))
            rc = self.random_choice(idx1.shape[1], numbers[i])        
            idx = np.concatenate((idx, idx1[:,rc]), axis = 1)
            id0 = np.concatenate((id0, i * np.ones(len(rc))))
        idr = np.random.randint(0,(len(Dataset.PERMS)-1) * self.permutate + 1, len(id0))
        self.indices = (np.concatenate((idr[np.newaxis,:], id0[np.newaxis,:],idx)).T).astype('uint16') 
        
    # get a test dataset:
    # train = Dataset(32)
    # test = train.detach_dataset()
    # now train uses only 90% of all bones and test the remaining 10%
    def detach_dataset(self, ratio = .1):
        test = copy.deepcopy(self)
        idx = int(len(self.fns) * (1-ratio))
        self.fns = self.fns[:idx]
        test.fns = test.fns[idx:]
        test.idx_fn = len(test.fns)
        if hasattr(self,'indices'):
            test.next_epoch()
        return test
       
    
    def _make_index():
        print('Intentando crear el archivo de índice...')
        if not os.path.exists(Dataset.PATH):
            os.makedirs(Dataset.PATH)
            print(f"Carpeta creada con éxito: {Dataset.PATH}")
    
        fns = os.listdir(Dataset.PATH)
        numbers = np.zeros((256, len(fns)), dtype='uint32')
        for i_fn, fn in tqdm.tqdm(enumerate(fns)):
            mask = np.load(os.path.join(Dataset.PATH, fn))['mask']
            h = np.histogram(mask, np.arange(257)-0.5)[0]
            numbers[:, i_fn] = np.cumsum(h[::-1])[::-1]
    
        np.savez_compressed(os.path.join(Dataset.PATH, Dataset.FN_INDEX), fns=fns, numbers=numbers)


if __name__ == "__main__":
    print("Llamando a Dataset directamente desde el bloque principal...")
    dataset = Dataset(32) 
