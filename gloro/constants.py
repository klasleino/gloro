from gloro.layers import Bias
from gloro.layers import InvertibleDownsampling
from gloro.layers import MinMax
from gloro.layers import Scaling
from gloro.relaxations.metrics import affinity_vra
from gloro.relaxations.metrics import affinity_vra_cat
from gloro.relaxations.metrics import affinity_vra_sparse
from gloro.relaxations.metrics import rtk_vra
from gloro.relaxations.metrics import rtk_vra_cat
from gloro.relaxations.metrics import rtk_vra_sparse
from gloro.relaxations.metrics import top_k_clean_acc
from gloro.relaxations.metrics import top_k_clean_acc_cat
from gloro.relaxations.metrics import top_k_clean_acc_sparse
from gloro.training.losses import Crossentropy
from gloro.training.losses import Trades
from gloro.training.metrics import clean_acc
from gloro.training.metrics import clean_acc_cat
from gloro.training.metrics import clean_acc_sparse
from gloro.training.metrics import vra
from gloro.training.metrics import vra_cat
from gloro.training.metrics import vra_sparse


GLORO_CUSTOM_OBJECTS = {
    'Crossentropy': Crossentropy,
    'Trades': Trades,
    'clean_acc': clean_acc,
    'clean_acc_cat': clean_acc_cat,
    'clean_acc_sparse': clean_acc_sparse,
    'vra': vra,
    'vra_cat': vra_cat,
    'vra_sparse':vra_sparse,

    'top_k_clean_acc': top_k_clean_acc,
    'top_k_clean_acc_cat': top_k_clean_acc_cat,
    'top_k_clean_acc_sparse': top_k_clean_acc_sparse,
    'rtk_vra': rtk_vra,
    'rtk_vra_cat': rtk_vra_cat,
    'rtk_vra_sparse': rtk_vra_sparse,
    'affinity_vra': affinity_vra,
    'affinity_vra_cat': affinity_vra_cat,
    'affinity_vra_sparse': affinity_vra_sparse,

    'MinMax': MinMax,
    'InvertibleDownsampling': InvertibleDownsampling,
    'Scaling': Scaling,
    'Bias': Bias,
}

EPS = 1e-9

MAIN_BRANCH = 'main'
RESIDUAL_BRANCH = 'residual'
SKIP_BRANCH = 'skip'
