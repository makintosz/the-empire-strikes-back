MODEL_ARCHITECTURE = {
    'input_shape': [48, 96, 2],

    'l1_filters': 16,
    'l1_size': 5,
    'l1_activation': 'relu',
    'l1_padding': 'valid',

    'l2_pooling': 2,

    'l3_filters': 32,
    'l3_size': 5,
    'l3_activation': 'relu',
    'l3_padding': 'valid',

    'l4_pooling': 2,

    'l5_dense': 32,
    'l5_activation': 'relu',

    'l6_dense': 32,
    'l6_activation': 'relu',
    
    'l7_dense': 1,
    'l7_activation': 'sigmoid'
}

PRED_THRESHOLD = 0.5
