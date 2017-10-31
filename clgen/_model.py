#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
CLgen model.
"""
import os
import progressbar
import re
import sys
import tarfile
import threading

from copy import deepcopy
from datetime import datetime
from labm8 import crypto
from labm8 import fs
from labm8 import jsonutil
from labm8 import lockfile
from labm8 import types
from prettytable import PrettyTable
from time import time
from typing import Iterator, List, Union

import clgen
from clgen import log
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
import tensorflow.contrib.seq2seq as seq2seq



# Default options used for model. Any values provided by the user will override
# these defaults.
DEFAULT_MODEL_OPTS = {
    "created": {
        "author": clgen.get_default_author(),
        "date": str(datetime.now()),
        "version": clgen.version(),
    },
    "architecture": {
        "model_type": "lstm",  # {lstm,rnn.gru}
        "rnn_size": 128,  # num nodes in layer
        "num_layers": 2,  # num layers
    },
    "train_opts": {
        "epochs": 10,
        "grad_clip": 5,
        "learning_rate": 2e-3,  # initial learning rate
        "lr_decay_rate": 5,  # % to reduce learning rate by per epoch
        "intermediate_checkpoints": True
    }
}


class ModelError(clgen.CLgenError):
    """
    Module level error
    """
    pass

def get_cell(model_type):
    cell_fn = {
        "lstm": rnn.BasicLSTMCell,
        "gru": rnn.GRUCell,
        "rnn": rnn.BasicRNNCell
    }.get(model_type, None)
    if cell_fn is None:
        raise clgen.UserError("Unrecognized model type")
    return cell_fn


class SeqEncoder(clgen.CLgenObject):
    """
    The encoder Part of the model
    """
    def __init__(self, model_type, rnn_size, num_layers, batch_size, vocab_size):
        self.cell_fn = get_cell(model_type)
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.vocab_size = vocab_size

    def encode(self, inputs, lengths):
        cells_lst = [self.cell_fn(self.rnn_size, state_is_tuple=True) for _ in range(self.num_layers)]
        cell = rnn.MultiRNNCell(cells_lst, state_is_tuple=True)

        self.initial_state = cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('encoder'):
            with tf.device('/cpu:0'):
                # Inputs 
                embedding = tf.get_variable('embedding', [self.vocab_size, self.rnn_size])
                inp = tf.nn.embedding_lookup(embedding, inputs)

            _, output_state = tf.nn.dynamic_rnn(cell, inp,
                sequence_length = lengths,
                initial_state = self.initial_state,
                dtype = tf.float32, swap_memory = True,
                time_major = False)

        return output_state


class Model(clgen.CLgenObject):
    """
    A CLgen Model.
    """
    def __init__(self, corpus: clgen.Corpus, **opts):
        """
        Instantiate model.

        Parameters
        ----------
        corpus : clgen.Corpus
            Corpus instance.
        **opts
            Training options.
        """
        assert(isinstance(corpus, clgen.Corpus))

        def _hash(corpus: clgen.Corpus, opts: dict) -> str:
            """ compute model hash """
            hashopts = deepcopy(opts)
            del hashopts["created"]
            del hashopts["train_opts"]["epochs"]
            return crypto.sha1_list(corpus.hash, *types.dict_values(hashopts))

        # Validate options
        for key in opts:
            if key not in DEFAULT_MODEL_OPTS:
                raise clgen.UserError(
                    "Unsupported model option '{}'. Valid keys: {}".format(
                        key, ','.join(sorted(DEFAULT_MODEL_OPTS.keys()))))

        # set properties
        self.opts = types.update(deepcopy(DEFAULT_MODEL_OPTS), opts)
        self.corpus = corpus
        self.hash = _hash(self.corpus, self.opts)
        self.cache = clgen.mkcache("model", self.hash)

        log.debug("model", self.hash)

        # validate metadata against cache, and restore stats
        self.stats = {
            "epoch_times": [],
            "epoch_costs": [],
            "epoch_batches": []
        }
        meta = deepcopy(self.to_json())
        if self.cache.get("META"):
            cached_meta = jsonutil.read_file(self.cache["META"])
            self.stats = cached_meta["stats"]  # restore stats

            if "created" in cached_meta:
                del cached_meta["created"]
            del meta["created"]

            if "created" in cached_meta["corpus"]:
                del cached_meta["corpus"]["created"]
            del meta["corpus"]["created"]

            if "stats" in cached_meta:
                del cached_meta["stats"]
            del meta["stats"]

            if "epochs" in cached_meta["train_opts"]:
                del cached_meta["train_opts"]["epochs"]
            del meta["train_opts"]["epochs"]

            if meta != cached_meta:
                log.error("Computed META:", jsonutil.format_json(meta))
                raise clgen.InternalError(
                    "metadata mismatch in model %s" % self.cache["META"])
        else:
            self._flush_meta()

    def _init_tensorflow(self, infer: bool=False) -> 'tf':
        """
        Deferred importing of tensorflow and initializing model for training
        or sampling.

        This is necessary for two reasons: first, the tensorflow graph is
        different for training and inference, so must be reset when switching
        between modes. Second, importing tensorflow takes a long time, so
        we only want to do it if we actually need to.

        Parameters
        ----------
        infer : bool
            If True, initialize model for inference. If False, initialize
            model for training.

        Returns
        -------
        module
            TensorFlow module.
        """
        # quiet tensorflow. See: https://github.com/tensorflow/tensorflow/issues/1258
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.cell_fn = {
            "lstm": rnn.BasicLSTMCell,
            "gru": rnn.GRUCell,
            "rnn": rnn.BasicRNNCell
        }.get(self.model_type, None)
        if self.cell_fn is None:
            raise clgen.UserError("Unrecognized model type")

        # reset the graph when switching between training and inference
        tf.reset_default_graph()

        # corpus info:
        batch_size = 1 if infer else self.corpus.batch_size
        seq_length = 1 if infer else self.corpus.seq_length
        vocab_size = self.corpus.vocab_size

        cells_lst = [self.cell_fn(self.rnn_size, state_is_tuple=True) for _ in range(self.num_layers)]
        self.cell = rnn.MultiRNNCell(cells_lst, state_is_tuple=True)

        with tf.device("/cpu:0"):
            # Inputs 
            self.encoder_input = tf.placeholder(tf.int32, [batch_size, seq_length])
            self.decoder_input = tf.placeholder(tf.int32, [batch_size, seq_length])
            self.target_weights = tf.placeholder(tf.int32, [batch_size, seq_length])
            self.lengths = tf.placeholder(tf.int32, [batch_size])

            self.q = tf.FIFOQueue(capacity=4,
                dtypes=[tf.int32, tf.int32, tf.int32, tf.int32],
                shapes=[tf.TensorShape([batch_size, seq_length]), 
                    tf.TensorShape([batch_size, seq_length]),
                    tf.TensorShape([batch_size, seq_length]),
                    tf.TensorShape([batch_size])])
            self.enqueue_op = self.q.enqueue((self.encoder_input, self.decoder_input, self.target_weights, self.lengths))

            next_example = self.q.dequeue()

            self.inputs = next_example[0]
            self.dec_inp = next_example[1]
            self.tweights = tf.to_float(next_example[2])
            self.lens = next_example[3]
        

        scope_name = 'rnnlm'
        with tf.variable_scope(scope_name):
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, vocab_size])
            softmax_b = tf.get_variable("softmax_b", [vocab_size])

            with tf.device("/cpu:0"):
                embedding_dec = tf.get_variable("embedding_dec", [vocab_size, self.rnn_size])
                dec_inp2 = tf.nn.embedding_lookup(embedding_dec, self.dec_inp)

        encoder = SeqEncoder(self.model_type, self.rnn_size, self.num_layers, batch_size, vocab_size)
        encoder_state = encoder.encode(self.inputs, self.lens)

        self.mean_latent, self.logvar_latent = encoder_to_latent(encoder_state, self.rnn_size, 32, self.num_layers, tf.float32)
        self.latent, self.KL_obj, self.KL_cost = sample(self.mean_latent, self.logvar_latent, 32)
        self.decoder_initial_state = latent_to_decoder(self.latent, self.rnn_size, 32, self.num_layers, tf.float32)


        decoder_initial_state2 = tuple([rnn.LSTMStateTuple(*single_layer_state) for single_layer_state in self.decoder_initial_state])

        helper = seq2seq.TrainingHelper(dec_inp2, self.lens, time_major=False)
        decoder = seq2seq.BasicDecoder(self.cell, helper, decoder_initial_state2, Dense(vocab_size))
        self.final_outputs, self.final_state = seq2seq.dynamic_decode(decoder, output_time_major=False, impute_finished=True, swap_memory=True, scope='rnnlm')

        self.final_out = self.final_outputs.rnn_output

        self.probs = tf.nn.softmax(self.final_out)
        self.cost = seq2seq.sequence_loss(self.final_out, self.inputs, self.tweights)

        self.learning_rate = tf.Variable(0.0, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost + self.KL_obj, tvars, aggregation_method = 2), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        return tf


    def _get_params_path(self, ckpt) -> str:
        """ return path to checkpoint closest to target num of epochs """
        paths = ckpt.all_model_checkpoint_paths
        batch_nums = [int(x.split('-')[-1]) for x in paths]
        epoch_nums = [int((x + 1) / (self.corpus.num_batches))
                      for x in batch_nums]

        closest = self.epochs
        closest_path = None
        for e, path in zip(epoch_nums, paths):
            diff = self.epochs - e
            if diff >= 0 and diff < closest:
                log.verbose("  cached checkpoint at epoch =", e, "diff =", diff)
                closest = diff
                closest_path = path

        return closest_path, paths

    def enqueue_x(self, coord, sess):
        with coord.stop_on_exception():
            while not coord.should_stop():
                x_enc, x_dec, w, l = self.corpus.next_batch()
                sess.run(self.enqueue_op, feed_dict={self.encoder_input: x_enc, self.decoder_input: x_dec, self.target_weights: w, self.lengths: l})

    def _locked_train(self) -> 'Model':
        tf = self._init_tensorflow(infer=False)

        # training options
        learning_rate = self.train_opts["learning_rate"]
        decay_rate = self.train_opts["lr_decay_rate"]

        # resume from prior checkpoint
        ckpt_path, ckpt_paths = None, None
        if self.checkpoint_path:
            # check that all necessary files exist
            assert(fs.isdir(self.checkpoint_path))
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
            assert(ckpt)
            assert(ckpt.model_checkpoint_path)
            ckpt_path, ckpt_paths = self._get_params_path(ckpt)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            # keep all checkpoints
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

            # restore model from closest checkpoint
            if ckpt_path:
                log.debug("restoring", ckpt_path)
                saver.restore(sess, ckpt_path)
                log.verbose("restored checkpoint {}".format(ckpt_path))

            # make sure we don't lose track of other checkpoints
            if ckpt_paths:
                saver.recover_last_checkpoints(ckpt_paths)

            coord = tf.train.Coordinator()
            self.corpus.create_batches()
            threading.Thread(target=self.enqueue_x, args=(coord, sess)).start()

            max_batch = self.epochs * self.corpus.num_batches

            # progress bar
            bar = progressbar.ProgressBar(max_value=max_batch)

            if sess.run(self.epoch) != self.epochs:
                log.info("training", self)

            for e in range(sess.run(self.epoch) + 1, self.epochs + 1):
                epoch_start = time()

                # decay and set learning rate
                new_learning_rate = learning_rate * (
                    (float(100 - decay_rate) / 100.0) ** (e - 1))
                sess.run(tf.assign(self.learning_rate, new_learning_rate))
                sess.run(tf.assign(self.epoch, e))

                for b in range(self.corpus.num_batches):
                    train_cost, _, state, _ = sess.run([self.cost, self.KL_cost, self.final_state, self.train_op])
                    # update progress bar
                    batch_num = (e - 1) * self.corpus.num_batches + b
                    bar.update(batch_num)

                save = self.opts["train_opts"]["intermediate_checkpoints"]
                save |= e == self.epochs  # always save on last epoch
                if save:
                    saver.save(sess, self.cache.keypath("model.ckpt"),
                               global_step=batch_num)

                    next_checkpoint = e * self.corpus.num_batches + b
                    max_epoch = self.epochs
                    log.verbose("\n{self} epoch {e} / {max_epoch}. "
                                "next checkpoint at batch {next_checkpoint}"
                                .format(**vars()))

                    # update training time
                    epoch_duration = time() - epoch_start
                    self.stats["epoch_costs"].append(float(train_cost))
                    self.stats["epoch_times"].append(epoch_duration)
                    self.stats["epoch_batches"].append(batch_num + 1)
                    self._flush_meta()
            coord.request_stop()
        return self

    def _flush_meta(self) -> None:
        jsonutil.write_file(self.cache.keypath("META"), self.to_json())

    def train(self) -> 'Model':
        """
        Train model.

        Returns
        -------
        Model
            self
        """
        with self.lock.acquire(replace_stale=True):
            return self._locked_train()

    @property
    def shorthash(self):
        return clgen._shorthash(self.hash, clgen.cachepath("model"))

    @property
    def lock(self) -> lockfile.LockFile:
        lockpath = self.cache.keypath("LOCK")
        return lockfile.LockFile(lockpath)

    @property
    def model_type(self) -> str:
        return self.opts["architecture"]["model_type"]

    @property
    def rnn_size(self) -> int:
        return self.opts["architecture"]["rnn_size"]

    @property
    def num_layers(self) -> int:
        return self.opts["architecture"]["num_layers"]

    @property
    def grad_clip(self) -> int:
        return self.train_opts["grad_clip"]

    @property
    def epochs(self) -> int:
        return self.train_opts["epochs"]

    @property
    def train_opts(self) -> dict:
        return self.opts["train_opts"]

    def __repr__(self) -> str:
        """
        String representation.
        """
        celltype = self.model_type.upper()
        return (f"model[{self.shorthash}]: " +
                f"{self.rnn_size}x{self.num_layers}x{self.epochs} {celltype}")

    def to_json(self) -> dict:
        d = deepcopy(self.opts)
        d["corpus"] = self.corpus.to_json()
        d["stats"] = self.stats
        return d

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, Model):
            return False
        return rhs.hash == self.hash

    def __ne__(self, rhs) -> bool:
        return not self.__eq__(rhs)

    @property
    def checkpoint_path(self) -> Union[str, None]:
        """
        Get path to most recemt checkpoint, if exists.

        Returns
        -------
        Union[str, None]
            Path to checkpoint, or None if no checkpoints.
        """
        if self.cache.get("checkpoint"):
            return self.cache.path
        else:
            return None

    @staticmethod
    def from_json(model_json: dict) -> 'Model':
        """
        Load model from JSON.

        Parameters
        ----------
        model_json : dict
            JSON specification.

        Returns
        -------
        Model
            Model instance.
        """
        assert(isinstance(model_json, dict))

        if "corpus" not in model_json:
            raise clgen.UserError("model JSON has no corpus entry")

        # create corpus and remove from JSON
        corpus = clgen.Corpus.from_json(model_json.pop("corpus"))

        if "stats" in model_json:  # ignore stats
            del model_json["stats"]

        return Model(corpus, **model_json)


def models() -> Iterator[Model]:
    """
    Iterate over all cached models.

    Returns
    -------
    Iterator[Model]
        An iterable over all cached models.
    """
    if fs.isdir(clgen.cachepath(), "model"):
        modeldirs = fs.ls(fs.path(clgen.cachepath(), "model"), abspaths=True)
        for modeldir in modeldirs:
            meta = jsonutil.read_file(fs.path(modeldir, "META"))
            model = Model.from_json(meta)
            yield model


def models_to_tab(*models: List[Model]) -> PrettyTable:
    """
    Pretty print a table of model stats.

    Parameters
    ----------
    models : List[Model]
        Models to tablify.

    Returns
    -------
    PrettyTable
        Formatted table for printing.
    """
    tab = PrettyTable([
        "model",
        "corpus",
        "trained",
        "type",
        "nodes",
        "epochs",
        "lr",
        "dr",
        "gc",
    ])

    tab.align['nodes'] = 'r'
    tab.sortby = "nodes"

    for model in models:
        meta = model.to_json()

        nodes = meta["architecture"]["rnn_size"]
        layers = meta["architecture"]["num_layers"]

        if "stats" in meta:
            num_epochs = len(meta["stats"]["epoch_costs"])
        else:
            num_epochs = 0

        if num_epochs >= meta["train_opts"]["epochs"]:
            trained = "Y"
        elif fs.isfile(fs.path(model.cache.path, "LOCK")):
            trained = f"WIP ({num_epochs}/{meta['train_opts']['epochs']})"
        elif num_epochs > 0:
            trained = f"{num_epochs}/{meta['train_opts']['epochs']}"
        else:
            trained = ""

        tab.add_row([
            model.shorthash,
            model.corpus.shorthash,
            trained,
            meta["architecture"]["model_type"],
            f'{nodes} x {layers}',
            meta["train_opts"]["epochs"],
            "{:.0e}".format(meta["train_opts"]["learning_rate"]),
            meta["train_opts"]["lr_decay_rate"],
            meta["train_opts"]["grad_clip"],
        ])

    return tab

def encoder_to_latent(encoder_state,
                      rnn_size,
                      latent_dim,
                      num_layers,
                      dtype=None):
    concat_state_size = num_layers * rnn_size * 2
    encoder_state = list(map(lambda state_tuple: tf.concat(state_tuple, axis=1), encoder_state))
    encoder_state = tf.concat(encoder_state, axis=1)
    with tf.variable_scope('encoder_to_latent'):
        w = tf.get_variable("w",[concat_state_size, 2 * latent_dim], dtype=dtype)
        b = tf.get_variable("b", [2 * latent_dim], dtype=dtype)
        mean_logvar = tf.nn.relu(tf.matmul(encoder_state, w) + b)
        mean, logvar = tf.split(mean_logvar, 2, 1)

    return mean, logvar


def latent_to_decoder(latent_vector,
                      rnn_size,
                      latent_dim,
                      num_layers,
                      dtype=None):

    concat_state_size = num_layers * rnn_size * 2
    with tf.variable_scope('latent_to_decoder'):
        w = tf.get_variable("w", [latent_dim, concat_state_size], dtype=dtype)
        b = tf.get_variable("b", [concat_state_size], dtype=dtype)
        decoder_initial_state = tf.nn.relu(tf.matmul(latent_vector, w) + b)
        decoder_initial_state = tuple(tf.split(decoder_initial_state, num_layers, 1))
        decoder_initial_state = tuple([tuple(tf.split(single_layer_state, 2, 1)) for single_layer_state in decoder_initial_state])

    return decoder_initial_state

def sample(means, logvars, latent_dim):
    noise = tf.random_normal(tf.shape(means))
    sample = means + tf.exp(0.5 * logvars) * noise
    kl_cost = -0.5 * (logvars - tf.square(means) - tf.exp(logvars) + 1.0)
    kl_ave = tf.reduce_mean(kl_cost, [0]) #mean of kl_cost over batches
    kl_obj = kl_cost = tf.reduce_sum(kl_ave)

    return sample, kl_obj, kl_cost #both kl_obj and kl_cost are scalar
