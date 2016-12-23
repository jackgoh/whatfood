nb_class = 100
n_folds = 2
learning_rate = 0.1
weights_decay = 0.0005
decay = 1e-6
momentum = 0.9
nb_epoch = 250

data_path = "../dataset/myfood10-227.hkl"
alexnet_weights_path = '../dataset/alexnet_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'

classes = ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
         'BananaLeafRice' , 'Bazhang' , 'BeefRendang' , 'BingkaUbi']
