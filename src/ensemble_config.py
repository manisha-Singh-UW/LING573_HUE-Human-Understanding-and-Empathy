seed = 573

use_embeddings = {'MiniLM'  : False,
                  'mpnet'   : False,
                  'roberta' : False,
                  'ada'     : True}

embedding2str = {'MiniLM'  : 'all-MiniLM-L6-v2',
                 'mpnet'   : 'all-mpnet-base-v2',
                 'roberta' : 'all-roberta-large-v1',
                 'ada'     : 'text-embedding-ada-002'}

use_models = {'NN'      : True,
              'sklearn' : True}

use_kernels = {'linear'  : False,
               'poly'    : True,
               'rbf'     : True,
               'sigmoid' : False}
