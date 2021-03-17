MODEL_ARCHITECTURE = {
    'input_shape': [48, 96, 2],

    'l1_filters': 8,
    'l1_size': 5,
    'l1_activation': 'relu',
    'l1_padding': 'same',

    'l2_pooling': 2,

    'l3_dense': 8,
    'l3_activation': 'relu',
    
    'l4_dense': 1,
    'l4_activation': 'sigmoid'
}

PRED_THRESHOLD = 0.5
