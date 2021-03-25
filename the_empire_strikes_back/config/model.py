MODEL_ARCHITECTURE = {
    'input_shape': [48, 96, 2],

    'l1_filters': 32,
    'l1_size': 5,
    'l1_activation': 'relu',
    'l1_padding': 'valid',

    'l2_pooling': 2,

    'l3_filters': 16,
    'l3_size': 5,
    'l3_activation': 'relu',
    'l3_padding': 'valid',

    'l4_pooling': 2,

    'l5_dense': 8,
    'l5_activation': 'relu',
    
    'l6_dense': 1,
    'l6_activation': 'sigmoid'
}

PRED_THRESHOLD = 0.4
