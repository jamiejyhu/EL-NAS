### Search Process
1. dataprocess.py => dataName.npz
2. train_search.py => genotype => write to => genotype.py(genotypeName)
3. train.py => --arch(genotypeName) => run

### Eval Process
1. train.py => --arch("indian3p") => --dataset("indian3p.npz") => run
2. train.py => --arch("pavia1p") => --dataset("pavia1p.npz") => run
3. train.py => --arch("houston3p") => --dataset("houston3p.npz") => run