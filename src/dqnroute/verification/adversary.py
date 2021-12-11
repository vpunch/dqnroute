import torch as tch
import numpy as np

from .edtsolver import EDTSolver, ProbsEmbs


class PGDAdversary:
    def __init__(self,
                 max_pertb_l:   float,
                 n_repeat:      int,
                 is_rand_start: bool,
                 is_euclid_l:   bool,
                 n_steps:       int,
                 stop_loss:     float,
                 is_minimize_l: bool,
                 step_size:     float,
                 verbose:       bool) -> None:
        self.max_pertb_l = max_pertb_l
        self.n_repeat = n_repeat
        self.is_rand_start = is_rand_start
        self.is_euclid_l = is_euclid_l
        self.n_steps = n_steps
        self.stop_loss = stop_loss
        self.is_minimize_l = is_minimize_l
        self.step_size = step_size
        self.verbose = verbose

    def perturb(self, edt_solver: EDTSolver) -> tuple[tch.Tensor, float]:
        # May change during execution
        max_pertb_l = self.max_pertb_l
        best_edt = -1

        embs_vec = self._get_embs_vec(edt_solver)
        if self.verbose:
            print('The beginning of perturbation')
            print(f'x = {embs_vec}')

        # We make several descents and choose the best result
        for repeat_n in range(self.n_repeat):
            if self.is_rand_start and repeat_n != self.n_repeat - 1:
                # Heuristic: at the last iteration, always start at the
                # center of the unit circle
                init_pertb = self._get_rand_pertb(embs_vec.numel(),
                                                  max_pertb_l)
            else:
                init_pertb = embs_vec * 0

            pertb, edt = self._get_descent(edt_solver,
                                           embs_vec,
                                           init_pertb,
                                           max_pertb_l)

            if self.verbose:
                print(f'Descent number: {repeat_n}')
                print(f'║Δx║ = {pertb}')
                print(f'EDT: {edt}')

            if edt > best_edt:  # always True on the first iteration
                best_edt = edt
                best_pertb = pertb

                # We got an acceptable result
                if edt > self.stop_loss:
                    # We want to find another solution that has a lower
                    # norm
                    if self.is_minimize_l:
                        pertb_l = self._calc_l(pertb)

                        if pertb_l < max_pertb_l:
                            max_pertb_l = pertb_l
                            best_edt = self.stop_loss
                    else:
                        break

        return embs_vec + best_pertb, best_edt

    def _get_embs_vec(self, edt_solver: EDTSolver) -> tch.Tensor:
        """Concatenate the embeddings needed to calculate the expected
        delivery time

        Embeddings are placed in the single vector as follows:
        diverter,
        neighbour on the same conveyor
        second neighbour,
        ... (repeat for each diverter)
        sink
        """

        dvtr_vecs = []

        for demb, nembs, iemb in edt_solver.probs_embs:
            dvtr_vecs.append(tch.cat((demb, tch.flatten(nembs))))

        dvtr_vecs.append(iemb)

        return tch.cat(dvtr_vecs)

    def _get_rand_pertb(self, pertb_len, max_l):
        """Get a random perturbation from a uniform distribution"""

        if self.is_euclid_l:
            # Get a random direction
            pertb = tch.randn(pertb_len, dtype=tch.float32)
            # Take a point from the surface of the sphere
            pertb = self._project(pertb, max_l)
            # Move the point randomly towards the center
            #pertb *= np.random.rand()
        else:
            # Values from a uniform distribution from -1 to 1
            pertb = tch.rand(pertb_len, dtype=tch.float32) * 2 - 1
            # To scale
            pertb *= max_l

        return pertb * np.random.rand()

    def _get_perturb(self, vec, grad):
        if self.is_euclid_l:
            vec_l = self._calc_l(vec)
            if vec_l:
                return vec / vec_l

            return vec * 0
        else:

            return vec.sign()

    def _project(self, vec, max_l):
        # clip
        if self.is_euclid_l:
            return self._normalize(vec) * max_l
        else:
            return vec.clamp(-max_l, max_l)

    def _calc_l(self, vec: tch.Tensor) -> tch.Tensor:
        return vec.norm(2 if self.is_euclid_l else np.infty)

    def _get_descent(self,
                     edt_solver:  EDTSolver,
                     embs_vec:    tch.Tensor,
                     pertb:       tch.Tensor,
                     max_pertb_l: float) -> tuple[tch.Tensor, float]:
        for _ in range(self.n_steps):
            adv_embs_vec = embs_vec + pertb
            # Activate gradient computation 
            adv_embs_vec.requires_grad_()

            edt = edt_solver.get_edt_val(self._split_embs_vec(
                    adv_embs_vec,
                    edt_solver.n_dvtrs))

            edt.backward()
            grad = adv_embs_vec.grad
            #print(f'grad = {grad}')

            edt = edt.item()
            if edt > self.stop_loss:
                break

            # Zero gradient
            if grad.norm() == 0:
                break

            # сходится лучше, если фиксировать шаг
            pertb_step = self.step_size * self._normalize(grad) * max_pertb_l
            pertb += pertb_step

            if self._calc_l(pertb) > max_pertb_l:
                pertb = self._project(pertb, max_pertb_l)

            #print(f'run: {_}')
            print(f'║Δx║ = {self._calc_l(pertb)}')
            print(edt)
            #print(f'EDT: {edt}')

        return pertb, edt

    def _split_embs_vec(self, embs_vec: tch.Tensor, n_dvtrs: int) -> ProbsEmbs:
        """Inverse function of _get_embs_vec"""

        emb_len = embs_vec.numel() // (n_dvtrs * 3 + 1)

        probs_embs = []
        iemb = embs_vec[-emb_len:]
        for i in range(n_dvtrs):
            d_i = i * emb_len * 3
            fn_i = d_i + emb_len
            sn_i = fn_i + emb_len
            nd_i = sn_i + emb_len

            demb = embs_vec[d_i:fn_i]
            nembs = tch.stack((embs_vec[fn_i:sn_i], embs_vec[sn_i:nd_i]))

            probs_embs.append((demb, nembs, iemb))

        return probs_embs
