
class Config:
    def __init__(self,task):
        if task == "Flare22":
            self.base_dir = 'data/2022_flare_10'
            self.save_dir = 'DHC_flare_2024'
            self.patch_size = (64, 128, 128)
            self.num_cls = 14
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 100
        else: # amos
            self.base_dir = './Datasets/amos22'
            self.save_dir = './amos_data'
            self.patch_size = (64, 128, 128)
            self.num_cls = 16
            self.num_channels = 1
            self.n_filters = 32
            self.early_stop_patience = 50