
from pathlib import Path
from keras.layers import Input #,Layer, Dense, Lambda
from invLayer import *


class InvNN_base():
    def __init__(self, in_dim, inv_mode=1, fn_weights=None):
        self.encoder = None
        self.decoder = None
        self.in_dim = in_dim
        self.fn_weights = fn_weights
        self.shuffle_func = {0: Shuffle, 1: InvDense, 2: InvDense_LU}[inv_mode]
    def build_encoder(self, verbose_summary=False, n_depths=20, **kwargs):
        assert self.encoder == None, 'ERROR: build encoder twice'
        self.layer_list = []
        for _ in range(2):
            self.layer_list.append([])
        x_in = Input(shape=(self.in_dim,))
        x = x_in

        self.split = SplitVector()
        self.couple = CoupleWrapper()
        self.concat = ConcatVector()

        for depth in range(n_depths):
            shuffle = self.shuffle_func()
            basicNN = build_NN(self.in_dim // 2, **kwargs)
            self.layer_list[0].append(shuffle)
            self.layer_list[1].append(basicNN)
            x = shuffle(x)
            x = self.split(x)
            x = self.couple(x, basicNN)
            x = self.concat(x)
        self.encoder = Model(x_in, x)

        if verbose_summary:
            self.encoder.summary()

        self.encoder.compile(
            loss=lambda y_true,y_pred: K.sum(0.5 * y_pred**2, 1),
            optimizer='adam'
        )
    def train(self, 
        train_data, val_data, 
        batch_size=128, n_epoches=30, 
        fn_weights='InvDense.weights',
        load_exist_weight=False,
    ):
        assert self.encoder != None, 'need to build() first'
        self.fn_weights = fn_weights

        self.checkpoint = ModelCheckpoint(
            filepath=fn_weights,
            monitor='val_loss',
            verbose=1,
            save_best_only=True
        )
        if load_exist_weight and Path(fn_weights).is_file():
            encoder.load_weights(fn_weights)
        self.encoder.fit(
            train_data, train_data,
            validation_data=(val_data, val_data),
            batch_size=batch_size,
            epochs=n_epoches,
            callbacks=[self.checkpoint]
        )
    def build_decoder(self):
        assert self.decoder == None, 'ERROR: build decoder twice'

        if self.fn_weights != None and Path(self.fn_weights).is_file():
            self.encoder.load_weights(self.fn_weights)
        else:
            assert False, 'fn_weights is None'

        set_backwardpass()
        x_in = Input(shape=(self.in_dim,))
        x = x_in
        for (shuffle, basicNN) in list(zip(*self.layer_list))[::-1]:
            x = self.concat.inverse()(x)
            x = self.couple.inverse()(x, basicNN)
            x = self.split.inverse()(x)
            x = shuffle.inverse()(x)
        self.decoder = Model(x_in, x)



