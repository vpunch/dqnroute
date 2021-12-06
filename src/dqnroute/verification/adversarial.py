import numpy as np
import torch

class PGDAdversary:
    def __init__(self,
                 max_pertb_l,
                 n_repeat,
                 is_rand_start,
                 is_inf_l,
                 n_steps,
                 stop_loss,
                 is_minimize_l
                 verbose):
        self.max_pertb_l = max_pertb_l
        self.n_repeat = n_repeat
        self.is_rand_start = is_rand_start
        self.is_inf_l = is_inf_l
        self.n_steps = n_steps
        self.stop_loss = stop_loss
        self.is_minimize_l = is_minimize_l
        self.verbose = verbose

    def perturb(self, edt_solver):
        # Может изменяться в процессе выполнения
        max_pertb_l = self.max_pertb_l
        best_edt = -1

        for repeat_n in range(self.n_repeat):
            embs_vec = self._get_embs_vec(edt_solver)

            if is_rand_start and repeat_n != self.n_repeat - 1:
                # Эвристика: на последней итерации всегда начинаем с
                # центра
                pertb = self._get_rand_pertb(embs_vec.numel(), max_pertb_l)
            else:
                pertb = embs_vec * 0

            edt = self._get_descent(edt_solver, embs_vec, pertb, max_pertb_l)
            if edt > best_edt:  # всегда True на первой итерации
                best_edt = edt
                best_pertb = pertb

                if edt > self.stop_loss:
                    if self.is_minimize_l:
                        pertb_l = self._calc_l(pertb)

                        if pertb_l < max_pertb_l:
                            max_pertb_l = pertb_l
                            best_edt = self.stop_loss
                    else:
                        # Последняя итерация
                        break

        return embs_vec + best_pertb

    def _get_embs_vec(self, edt_solver):
        dvtr_vecs = []

        for demb, nembs, iemb in edt_solver.probs_embs:
            dvtr_vecs.append(torch.cat((demb, *nembs), 1))

        dvtr_vecs.append(iemb)

        return torch.cat(dvtr_vecs, 1)

    def _get_rand_pertb(self, pertb_len, max_l):
        if self.is_inf_l:
            # Равномерное распределение от -1 до 1
            pertb = torch.rand(1, pertb_len, dtype=torch.float32) * 2 - 1
            # Масштабируем
            pertb *= max_l
        else:
            # Получаем случайное направление
            pertb = torch.randn(1, pertb_len, dtype=torch.float32)
            # Берем точку на поверхности сферы. Она имеет равномерное
            # распределение.
            pertb = self._project(pertb, max_l)
            # Перемещаем точку случайно по направлению к центру
            pertb *= np.random.rand()

        return pertb

    def _normalize(self, vec):
        vec_l = self._calc_l(vec)
        if vec_l:
            return x / vec_l

        return vec * 0

    def _project(self, vec, max_l):
        return self._normalize(vec) * max_l

    def _calc_l(self, vec):
        return vec.norm(np.infty if self.is_inf_l else 2)

    def _get_descent(self, edt_solver, embs_vec, pertb, max_pertb_l):
        for _ in range(self.n_steps):
            adv_embs_vec = embs_vec + pertb
            edt = edt_solver.get_edt_val(self._split_embs_vec(
                    adv_embs_vec,
                    edt_solver.n_dvtrs))

            edt.backward()
            grad = adv_embs_vec.grad

            edt = edt.item()
            if edt > self.stop_loss:
                return edt

            if grad.norm() == 0:
                return edt

            pertb_step = self.step_size * grad
            pertb += pertb_step

            if self._calc_l(pertb) > max_pertb_l:
                pertb = self._project(pertb, max_pertb_l)

        return pertb

    def _split_embs_vec(self, embs_vec, n_dvtrs):
        emb_len = embs_vec.numel() // (n_dvtrs * 3 + 1)

        q_val_embs = []
        iembs = embs_vec[-self.emb_len:] 
        for i in range(n_dvtrs):
            d_i = i * emb_len * 3
            fn_i = d_i + emb_len
            sn_i = fn_i + emb_len
            nd_i = sn_i + emb_len

            demb = embs_vec[d_i:fn_i]
            nembs = (embs_vec[fn_i:sn_i], embs_vec[sn_i:nd_i])

            q_val_embs.append((demb, nembs, iemb))

        return q_val_embs
