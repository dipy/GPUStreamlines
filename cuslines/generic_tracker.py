import logging
import numpy as np
from tqdm import tqdm
from trx.trx_file_memmap import TrxFile
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.tracking.streamlinespeed import compress_streamlines
from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import Tractogram

logger = logging.getLogger("GPUStreamlines")


class GenericTracker:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def set_compression_parameters(self, pos_dtype=np.float32, linearize=False, tol_error=0.1, max_segment_length=10):
        """
        Set compression parameters to compress generated streamlines.
        Only works with TRX.
        
        Parameters
        ----------
        pos_dtype : dtype, optional
        Data type to use for the positions of the streamlines.
        Default: np.float32

        linearize : bool, optional
        Whether to linearize the streamlines using [1].
        Default: False

        tol_error : float, optional
        If linearize is true, tolerance error in mm.
        Default: 0.1

        max_segment_length : float, optional
        If linearize is true, maximum length in mm of any given segment produced by the compression.
        Default: 10
    
        References
        ----------
        [1] Caroline Presseau, Pierre-Marc Jodoin, Jean-Christophe Houde, and Maxime Descoteaux.
            A new compression format for fiber tracking datasets.
            NeuroImage, 109:73-83, 2015. URL: 10.1016/j.neuroimage.2014.12.058
        """
        self.pos_dtype = pos_dtype
        self.linearize = linearize
        self.tol_error = tol_error
        self.max_segment_length = max_segment_length


    def _ngpus(self):
        return getattr(self, "ngpus", 1)

    def _pos_dtype(self):
        return getattr(self, "pos_dtype", np.float16)

    def _linearize(self):
        return getattr(self, "linearize", False)

    def _tol_error(self):
        return getattr(self, "tol_error", 0.1)

    def _max_segment_length(self):
        return getattr(self, "max_segment_length", np.inf)

    def _divide_chunks(self, seeds):
        global_chunk_sz = self.chunk_size * self._ngpus()
        nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz
        return global_chunk_sz, nchunks

    def generate_sft(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)
        buffer_size = 0
        generators = []

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(nchunks):
                self.seed_propagator.propagate(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                )
                buffer_size += self.seed_propagator.get_buffer_size()
                generators.append(self.seed_propagator.as_generator())
                pbar.update(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz].shape[0]
                )
        array_sequence = ArraySequence(
            (item for gen in generators for item in gen), buffer_size
        )
        return StatefulTractogram(array_sequence, ref_img, Space.VOX)

    def generate_trx(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)

        # Will resize by a factor of 2 if these are exceeded
        sl_len_guess = 100
        sl_per_seed_guess = 2
        n_sls_guess = sl_per_seed_guess * seeds.shape[0]

        # trx files use memory mapping
        trx_reference = TrxFile(reference=ref_img)
        trx_reference.streamlines._data = trx_reference.streamlines._data.astype(
            self._pos_dtype()
        )
        trx_reference.streamlines._offsets = trx_reference.streamlines._offsets.astype(
            np.uint64
        )

        trx_file = TrxFile(
            nb_streamlines=n_sls_guess,
            nb_vertices=n_sls_guess * sl_len_guess,
            init_as=trx_reference,
        )
        offsets_idx = 0
        sls_data_idx = 0

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(int(nchunks)):
                self.seed_propagator.propagate(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                )
                tractogram = Tractogram(
                    self.seed_propagator.as_array_sequence(),
                    affine_to_rasmm=ref_img.affine,
                )
                if len(tractogram) == 0:
                    continue

                tractogram.to_world()
                sls = tractogram.streamlines

                if self._linearize():
                    sls = ArraySequence(compress_streamlines(
                        sls,
                        tol_error=self._tol_error(),
                        max_segment_length=self._max_segment_length(),
                    ))
                sls._data = sls._data.astype(self._pos_dtype())

                new_offsets_idx = offsets_idx + len(sls._offsets)
                new_sls_data_idx = sls_data_idx + len(sls._data)

                if (
                    new_offsets_idx > trx_file.header["NB_STREAMLINES"]
                    or new_sls_data_idx > trx_file.header["NB_VERTICES"]
                ):
                    logger.info("TRX resizing...")
                    trx_file.resize(
                        nb_streamlines=new_offsets_idx * 2,
                        nb_vertices=new_sls_data_idx * 2,
                    )

                # TRX uses memmaps here
                trx_file.streamlines._data[sls_data_idx:new_sls_data_idx] = sls._data
                trx_file.streamlines._offsets[offsets_idx:new_offsets_idx] = (
                    sls_data_idx + sls._offsets
                )
                trx_file.streamlines._lengths[offsets_idx:new_offsets_idx] = (
                    sls._lengths
                )

                offsets_idx = new_offsets_idx
                sls_data_idx = new_sls_data_idx
                pbar.update(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz].shape[0]
                )
        trx_file.resize()

        return trx_file
