"""Research model implementations preserved from commit b995ba3."""

import torch
from torch import nn
import torch.nn.functional as F
import nemo.collections.asr as nemo_asr

class DistilEncDecCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    def __init__(self, cfg, trainer, teacher_model, use_logit_distillation=True, kd_alpha=0.1, kd_temperature=1.0, use_layerwise_distillation=False, layer_kd_alpha=1.0):
        super().__init__(cfg=cfg, trainer=trainer)
        self.teacher = teacher_model
        self.use_logit_distillation = use_logit_distillation
        self.kd_alpha = kd_alpha
        self.temperature = kd_temperature
        self.use_layerwise_distillation = use_layerwise_distillation
        self.layer_kd_alpha = layer_kd_alpha

        self.layer_proj = None

    def _init_layer_proj(self, stu_feat: torch.Tensor, tch_feat: torch.Tensor):
        """
        stu_feat: (B, H_s, T_s), tch_feat: (B, H_t, T_t)
        """
        H_s = stu_feat.size(1)
        H_t = tch_feat.size(1)
        # student→teacher projection
        self.layer_proj = nn.Linear(H_s, H_t).to(stu_feat.device)

    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None):
        """
        Forward pass of the model.

        Args:
            input_signal: Tensor that represents a batch of raw audio signals,
                of shape [B, T]. T here represents timesteps, with 1 second of audio represented as
                `self.sample_rate` number of floating point values.
            input_signal_length: Vector of length B, that contains the individual lengths of the audio
                sequences.
            processed_signal: Tensor that represents a batch of processed audio signals,
                of shape (B, D, T) that has undergone processing via some DALI preprocessor.
            processed_signal_length: Vector of length B, that contains the individual lengths of the
                processed audio sequences.

        Returns:
            A tuple of 3 elements -
            1) The log probabilities tensor of shape [B, T, D].
            2) The lengths of the acoustic sequence after propagation through the encoder, of shape [B].
            3) The greedy token predictions of the model of shape [B, T] (via argmax)
        """
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]
        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    def training_step(self, batch, batch_idx):

        #    (signal, signal_length, transcript, transcript_length)
        signal, signal_length, transcript, transcript_length = batch

        # 2) student forward
        log_probs, encoded_len, _ = self.forward(
            input_signal=signal, input_signal_length=signal_length
        )

        ctc_loss = self.loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_length,
        )

        # logit distillation loss
        if self.use_logit_distillation:
            with torch.no_grad():
                tch_log_probs, tch_encoded_len, tch_logits = self.teacher.forward(
                    input_signal=signal,
                    input_signal_length=signal_length,
                )
            T = self.temperature
            stu_logp = F.log_softmax(log_probs / T, dim=-1)
            tch_p    = F.softmax(tch_log_probs  / T, dim=-1)
            logit_kd_loss  = F.kl_div(stu_logp, tch_p, reduction="batchmean") * (T*T)
            self.log("train_kd_loss", logit_kd_loss, prog_bar=False, on_step=True, on_epoch=True)
        else:
            logit_kd_losskd_loss = torch.tensor(0.0, device=log_probs.device)

        # layerwise distillation loss
        layer_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_layerwise_distillation:
            # 1) raw waveform → feature
            proc_signal, proc_length = self.preprocessor(
                input_signal=signal,
                length=signal_length,
            )

            stu_feat, _ = self.encoder(
                audio_signal=proc_signal,
                length=proc_length,
            ) # torch.Size([32, 88, 405])
            with torch.no_grad():

                proc_signal_t, proc_length_t = self.teacher.preprocessor(
                    input_signal=signal,
                    length=signal_length,
                )
                tch_feat, _ = self.teacher.encoder(
                    audio_signal=proc_signal_t,
                    length=proc_length_t,
                ) # torch.Size([32, 176, 405])

            if self.layer_proj is None:
                self._init_layer_proj(stu_feat, tch_feat)
            B, H_s, T_s = stu_feat.size()               # (B, H_s, T_s) torch.Size([32, 88, 405])
            stu_feat = stu_feat.transpose(1, 2)         # (B, T_s, H_s) torch.Size([32, 405, 88])
            stu_flat = stu_feat.reshape(-1, H_s)        # (B*T_s, H_s)  torch.Size([12960, 88])
            proj_flat = self.layer_proj(stu_flat)       # (B*T_s, H_t)  torch.Size([12960, 176])
            stu_proj = proj_flat.reshape(B, T_s, -1)    # (B, T_s, H_t) torch.Size([32, 405, 176])
            stu_proj = stu_proj.transpose(1, 2)         # (B, H_t, T_s) torch.Size([32, 176, 405])

            # layer_loss = F.mse_loss(stu_aligned, tch_feat)
            layer_loss = F.mse_loss(stu_proj, tch_feat)
            self.log("train_layer_kd_loss", layer_loss, prog_bar=False, on_step=True, on_epoch=True)
        else:
            layer_loss = torch.tensor(0.0, device=log_probs.device)

        loss = ctc_loss\
             + (self.kd_alpha * logit_kd_loss if self.use_logit_distillation else 0.0)\
             + (self.layer_kd_alpha * layer_loss if self.use_layerwise_distillation else 0.0)

        # logging
        self.log("train_loss",     loss,     prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ctc_loss", ctc_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

class DiffKDModule(nn.Module):
    'Baseline diffusion distillation with a teacher autoencoder and iterative denoiser.'
    def __init__(self, cfg):
        super().__init__()

        self.steps       = cfg.get("diffusion_steps", 9)
        self.teacher_dim = cfg["teacher_dim"]
        self.student_dim = cfg["student_dim"]

        self.latent_dim  = cfg.get("latent_dim", min(self.teacher_dim, self.student_dim))

        #   - encoder: (B, teacher_dim, T) → (B, latent_dim, T)
        #   - decoder: (B, latent_dim, T) → (B, teacher_dim, T)
        self.encoder = nn.Conv1d(self.teacher_dim, self.latent_dim, kernel_size=1)
        self.decoder = nn.Conv1d(self.latent_dim, self.teacher_dim, kernel_size=1)

        #   (B, student_dim, T) → (B, latent_dim, T)
        self.proj    = nn.Conv1d(self.student_dim, self.latent_dim, kernel_size=1)

        self.denoiser = nn.Sequential(
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.latent_dim, self.latent_dim, kernel_size=3, padding=1),
        )

        self.recon_loss   = nn.MSELoss()
        self.distill_loss = nn.MSELoss()

    def forward(self, stu_feat, tch_feat, sampling_steps=None):
        """
        Args:
          stu_feat: Tensor [B, T, student_dim]
          tch_feat: Tensor [B, T, teacher_dim]

        Returns:
          diffkd_loss: scalar tensor = AE loss + distill loss
        """

        stu_feat = stu_feat.permute(0, 2, 1)  # [B, T, student_dim] → [B, student_dim, T]
        tch_feat = tch_feat.permute(0, 2, 1)  # [B, T, teacher_dim] → [B, teacher_dim, T]
        z_t = self.encoder(tch_feat).detach()            # [B, latent_dim, T]
        rec = self.decoder(z_t)                          # [B, teacher_dim, T]
        ae_loss = self.recon_loss(rec, tch_feat)         # Eq.(6)

        z_s = self.proj(stu_feat)                        # [B, latent_dim, T]

        x = z_s
        for _ in range(self.steps):
            pred_noise = self.denoiser(x)
            x = x - pred_noise / self.steps
        denoised = x                                     # ˆZ(stu)

        diffkd_loss = self.distill_loss(denoised, z_t)   # Eq.(7)

        return ae_loss + diffkd_loss

# ─────────────────────────────

class TeacherAutoEncoder(nn.Module):
    """Teacher feature (B, C_t, T) → latent (B, L, T) → recon (B, C_t, T)"""
    def __init__(self, teacher_dim: int, latent_dim: int):
        super().__init__()
        self.enc = nn.Conv1d(teacher_dim, latent_dim, kernel_size=1)
        self.dec = nn.Conv1d(latent_dim, teacher_dim, kernel_size=1)

    @torch.no_grad()
    def encode_nograd(self, x_ct):  # x_ct: (B, C_t, T)
        return self.enc(x_ct)

    def forward(self, x_ct):
        z_t = self.enc(x_ct)
        rec = self.dec(z_t)
        return z_t, rec

class StudentProjector(nn.Module):
    """Student feature (B, C_s, T) → latent (B, L, T)"""
    def __init__(self, student_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Conv1d(student_dim, latent_dim, kernel_size=1)

    def forward(self, x_cs):         # (B, C_s, T)
        return self.proj(x_cs)

class NoiseAdapter(nn.Module):
    'Predict a mixing coefficient in [0, 1] and mix student latents with Gaussian noise.'
    def __init__(self, latent_dim: int):
        super().__init__()
        self.gamma_head = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, z_latent):
        gamma = self.gamma_head(z_latent)            # (B,1,T), ∈(0,1)
        eps   = torch.randn_like(z_latent)           # (B,L,T)
        z_noisy = gamma * z_latent + (1.0 - gamma) * eps
        return z_noisy, gamma

class SimpleDenoiser(nn.Module):
    'A 1D CNN denoiser that preserves the latent dimension.'
    def __init__(self, latent_dim: int, steps: int = 5):
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, z_in):         # (B,L,T)
        x = z_in
        for _ in range(self.steps):
            pred_noise = self.net(x)
            x = x - pred_noise / self.steps
        return x                      # denoised latent

class FMLatent(nn.Module):
    'Wrap flow matching in latent space; convert (B, C, T) to the internal (B, T, C) layout.'
    def __init__(self, latent_dim: int, flow_cfg: dict):
        super().__init__()
        flow_cfg = dict(flow_cfg or {})
        flow_cfg.setdefault("student_dim", latent_dim)
        flow_cfg.setdefault("teacher_dim", latent_dim)
        flow_cfg.setdefault("shape_transform", "identity")
        flow_cfg.setdefault("meta_encoder_type", flow_cfg.get("meta_encoder_type", "mlp"))
        flow_cfg.setdefault("training_sampling", flow_cfg.get("training_sampling", 8))
        self.fm = FlowMatchingModule(flow_cfg)

        self.default_steps = int(flow_cfg.get("training_sampling", 8))

    @staticmethod
    def _to_BT_C(x_bct):  # (B,C,T) -> (B,T,C)
        return x_bct.transpose(1, 2)

    def forward(self, s_latent_bct, t_latent_bct, steps: int | None = None):
        """
        s_latent_bct/t_latent_bct: (B,L,T)
        returns: fm_loss (scalar), s_out_bct: (B,L,T)
        """
        s_btC = self._to_BT_C(s_latent_bct)
        t_btC = self._to_BT_C(t_latent_bct)
        layer_steps = int(steps or self.default_steps)
        fm_loss, s_out_btC = self.fm(s_btC, t_f=t_btC, layer_sampling_step=layer_steps, layer_id=None)
        s_out_bct = s_out_btC.transpose(1, 2)
        if not isinstance(fm_loss, torch.Tensor):
            fm_loss = torch.as_tensor(fm_loss, device=s_latent_bct.device, dtype=s_latent_bct.dtype)
        return fm_loss, s_out_bct
# ─────────────────────────────

class DistilFlowMatchingCTCModelBPE(nemo_asr.models.EncDecCTCModelBPE):
    'Historical reconstruction variants. See docs/reproducibility.md for the exact loss paths. Losses are summed over layers.'
    def __init__(
        self,
        cfg,
        trainer,
        teacher_model,

        use_ctc=True,
        use_logit_distillation=True,
        kd_alpha=0.1,
        kd_temperature=1.0,
        use_layerwise_distillation=False,
        layer_kd_alpha=1.0,

        version: int = 1,                           # ★ 1~5
        kd_loss_type: str = "mse",

        student_dim: int = 88,
        teacher_dim: int = 176,
        latent_dim: int = 96,

        use_flow_matching=False, flow_cfg=None,
        use_diffkd=False, diffkd_cfg=None,
    ):
        super().__init__(cfg=cfg, trainer=trainer)

        diffusion_steps = diffkd_cfg.get("diffusion_steps", 9) if diffkd_cfg is not None else 9
        self.teacher = teacher_model
        self.version = int(version)
        assert self.version in {1,2,3,4,5,6,7,8}, 'Version must be between 1 and 8.'

        self.use_ctc = use_ctc
        self.use_logit_distillation = use_logit_distillation
        self.kd_alpha = kd_alpha
        self.temperature = kd_temperature
        self.use_layerwise_distillation = use_layerwise_distillation
        self.layer_kd_alpha = layer_kd_alpha
        self.student_dim = student_dim
        self.teacher_dim = teacher_dim
        self.latent_dim  = latent_dim

        self.recon_crit = nn.MSELoss()
        self.kd_crit = nn.L1Loss() if kd_loss_type == "l1" else nn.MSELoss()

        self.tae   = TeacherAutoEncoder(teacher_dim=self.teacher_dim, latent_dim=self.latent_dim)
        self.sproj = StudentProjector(student_dim=self.student_dim, latent_dim=self.latent_dim)
        self.adapter = NoiseAdapter(latent_dim=self.latent_dim)
        self.denoiser = SimpleDenoiser(latent_dim=self.latent_dim, steps=diffusion_steps)
        self.fm_latent = FMLatent(latent_dim=self.latent_dim, flow_cfg=flow_cfg or {})
        self.fm_latent_2 = FMLatent(latent_dim=self.latent_dim, flow_cfg=flow_cfg or {})

        self.use_flow_matching = use_flow_matching
        self.use_diffkd = use_diffkd
        self.flow_matching = None
        if self.use_flow_matching:
            assert flow_cfg is not None
            self.flow_cfg = flow_cfg
            self.sampling_steps_per_layer = flow_cfg.get("sampling_steps_per_layer", None)
            assert self.sampling_steps_per_layer is None or len(self.sampling_steps_per_layer) == len(self.encoder.layers)
            self.flow_matching = FlowMatchingModule(flow_cfg)

        if self.use_diffkd:
            assert diffkd_cfg is not None
            self.diffkd = DiffKDModule(diffkd_cfg)

        self.stu_feats, self.tch_feats = [], []

        assert len(self.teacher.encoder.layers) == len(self.encoder.layers), 'Student and teacher must have the same number of layers.'
        for layer in self.encoder.layers:
            layer.register_forward_hook(self._capture_stu_feat)
        for layer in self.teacher.encoder.layers:
            layer.register_forward_hook(self._capture_tch_feat)

    def _capture_stu_feat(self, module, inp, out):  # out: (B, Hs, T)
        self.stu_feats.append(out)

    def _capture_tch_feat(self, module, inp, out):  # out: (B, Ht, T)
        self.tch_feats.append(out)

    @staticmethod
    def _BHT_to_BCT(x_bht):
        return x_bht

    @staticmethod
    def _BHT_to_BTH(x_bht):
        return x_bht.transpose(1, 2)

    def forward(self, input_signal=None, input_signal_length=None,
                processed_signal=None, processed_signal_length=None):

        self.stu_feats.clear()
        self.tch_feats.clear()

        has_input = input_signal is not None and input_signal_length is not None
        has_processed = processed_signal is not None and processed_signal_length is not None
        if (has_input ^ has_processed) is False:
            raise ValueError('Provide exactly one input pair: raw audio or processed features.')
        if not has_processed:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal, length=input_signal_length
            )
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        enc_out_s, enc_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)

        with torch.no_grad():
            proc_t, len_t = self.teacher.preprocessor(input_signal=input_signal, length=input_signal_length)
            _ = self.teacher.encoder(audio_signal=proc_t, length=len_t)

        log_probs = self.decoder(encoder_output=enc_out_s)
        greedy = log_probs.argmax(dim=-1)

        if self.training:
            dummy = torch.tensor(0.0, device=enc_out_s.device)
            return log_probs, enc_len, greedy, dummy, enc_out_s
        else:
            return log_probs, enc_len, greedy

    def _compute_v_losses_one_layer(self, s_bht, t_bht):
        'Compute reconstruction and variant-specific distillation losses for one pair of layer features.'
        # (B,C,T)
        s_bct = self._BHT_to_BTH(s_bht)
        t_bct = self._BHT_to_BTH(t_bht)

        z_t, t_rec = self.tae(t_bct)                 # (B,L,T), (B,Ct,T)
        z_t = z_t.detach()
        recon_loss = self.recon_crit(t_rec, t_bct)

        # Student proj: z_s
        z_s = self.sproj(s_bct)                      # (B,L,T)

        out = {
            "recon_loss": recon_loss,
            "kd_loss_pre": torch.zeros((), device=s_bct.device),
            "fm_loss_pre": torch.zeros((), device=s_bct.device),
            "kd_loss_post": torch.zeros((), device=s_bct.device),
            "fm_loss_post": torch.zeros((), device=s_bct.device),
        }

        if self.version == 1:
            # AE + KD
            out["kd_loss_pre"] = self.kd_crit(z_s, z_t)

        elif self.version == 2:

            fm_loss, _ = self.fm_latent(z_s, z_t)
            out["fm_loss_pre"] = fm_loss

        elif self.version == 3:
            # AE + NoiseAdapter + Diffusion + KD
            z_noisy, _ = self.adapter(z_s)
            z_deno = self.denoiser(z_noisy)
            out["kd_loss_post"] = self.kd_crit(z_deno, z_t)

        elif self.version == 4:

            fm_loss_pre, _ = self.fm_latent(z_s, z_t)
            z_noisy, _ = self.adapter(z_s)
            z_deno = self.denoiser(z_noisy)
            kd_loss_post = self.kd_crit(z_deno, z_t)
            out["fm_loss_pre"]  = fm_loss_pre
            out["kd_loss_post"] = kd_loss_post

        elif self.version == 5:

            z_noisy, _ = self.adapter(z_s)
            z_deno = self.denoiser(z_noisy)
            fm_loss, _ = self.fm_latent(z_deno, z_t)
            out["fm_loss_post"] = fm_loss

        elif self.version == 6:
            # AE + FM(pre) + NoiseAdapter + Diffusion + FM(post)
            fm_loss_pre, z_s_aligned = self.fm_latent(z_s, z_t)
            z_noisy, _ = self.adapter(z_s_aligned)
            z_deno = self.denoiser(z_noisy)
            fm_loss_post, _ = self.fm_latent_2(z_deno, z_t)
            out["fm_loss_pre"]  = fm_loss_pre
            out["fm_loss_post"] = fm_loss_post
        elif self.version == 7:
            # AE + FM(pre) + NoiseAdapter + Diffusion + FM(post)
            fm_loss_pre, _ = self.fm_latent(z_s, z_t)
            z_noisy, _ = self.adapter(z_s)
            z_deno = self.denoiser(z_noisy)
            fm_loss_post, _ = self.fm_latent_2(z_deno, z_t)
            out["fm_loss_pre"]  = fm_loss_pre
            out["fm_loss_post"] = fm_loss_post
        elif self.version == 8:

            fm_loss_pre, z_s_aligned = self.fm_latent(z_s, z_t)
            z_noisy, _ = self.adapter(z_s_aligned)
            z_deno = self.denoiser(z_noisy)
            kd_loss_post = self.kd_crit(z_deno, z_t)
            out["fm_loss_pre"]  = fm_loss_pre
            out["kd_loss_post"] = kd_loss_post

        return out

    def training_step(self, batch, batch_idx):
        signal, sig_len, transcript, transcript_len = batch

        log_probs, enc_len, greedy_preds, _dummy, enc_out = self.forward(
            input_signal=signal, input_signal_length=sig_len
        )

        # 2) CTC loss
        if self.use_ctc:
            ctc_loss = self.loss(
                log_probs=log_probs,
                targets=transcript,
                input_lengths=enc_len,
                target_lengths=transcript_len,
            )
        else:
            ctc_loss = torch.tensor(0.0, device=log_probs.device)

        # 3) Logit KD
        if self.use_logit_distillation:
            with torch.no_grad():
                tch_logp = self.teacher.decoder(encoder_output=self.tch_feats[-1].permute(0, 2, 1))
                tch_p = F.softmax(tch_logp / self.temperature, dim=-1)
            stu_logp = F.log_softmax(log_probs / self.temperature, dim=-1)
            logit_kd_loss = F.kl_div(stu_logp, tch_p, reduction="batchmean") * (self.temperature ** 2)
        else:
            logit_kd_loss = torch.tensor(0.0, device=log_probs.device)

        layer_kd_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_layerwise_distillation:
            for s, t in zip(self.stu_feats, self.tch_feats):
                B, Hs, T = s.shape
                Ht = t.size(1)
                s_flat = s.transpose(1,2).reshape(-1, Hs)
                p_flat = nn.Linear(self.student_dim, self.teacher_dim).to(s.device)(s_flat)
                s_proj = p_flat.reshape(B, T, Ht).transpose(1,2)
                layer_kd_loss = layer_kd_loss + F.mse_loss(s_proj, t)
            layer_kd_loss = layer_kd_loss / max(1, len(self.stu_feats))

        recon_sum = kd_pre_sum = fm_pre_sum = kd_post_sum = fm_post_sum = torch.zeros((), device=log_probs.device)
        L = max(1, len(self.stu_feats))
        for s, t in zip(self.stu_feats, self.tch_feats):
            losses = self._compute_v_losses_one_layer(s, t)
            recon_sum   = recon_sum   + losses["recon_loss"]
            kd_pre_sum  = kd_pre_sum  + losses["kd_loss_pre"]
            fm_pre_sum  = fm_pre_sum  + losses["fm_loss_pre"]
            kd_post_sum = kd_post_sum + losses["kd_loss_post"]
            fm_post_sum = fm_post_sum + losses["fm_loss_post"]

        # kd_loss_pre  = kd_pre_sum  / L
        # fm_loss_pre  = fm_pre_sum  / L
        # kd_loss_post = kd_post_sum / L
        # fm_loss_post = fm_post_sum / L
        recon_loss   = recon_sum
        kd_loss_pre  = kd_pre_sum
        fm_loss_pre  = fm_pre_sum
        kd_loss_post = kd_post_sum
        fm_loss_post = fm_post_sum

        diffkd_loss = torch.tensor(0.0, device=log_probs.device)
        if self.use_diffkd:
            for s, t in zip(self.stu_feats, self.tch_feats):
                diffkd_loss = diffkd_loss + self.diffkd(s, t)
            diffkd_loss = diffkd_loss / L

        total_loss = (
            ctc_loss
            + self.kd_alpha * logit_kd_loss
            + self.layer_kd_alpha * layer_kd_loss
            + recon_loss
            + kd_loss_pre + kd_loss_post
            + fm_loss_pre + fm_loss_post
            + diffkd_loss
        )

        self.log("loss/ctc", ctc_loss, on_step=True, on_epoch=True)
        self.log("loss/logit_kd", logit_kd_loss, on_step=True, on_epoch=True)
        self.log("loss/layer_kd", layer_kd_loss, on_step=True, on_epoch=True)

        self.log("v/recon",   recon_loss,   on_step=True, on_epoch=True)
        self.log("v/kd_pre",  kd_loss_pre,  on_step=True, on_epoch=True)
        self.log("v/fm_pre",  fm_loss_pre,  on_step=True, on_epoch=True)
        self.log("v/kd_post", kd_loss_post, on_step=True, on_epoch=True)
        self.log("v/fm_post", fm_loss_post, on_step=True, on_epoch=True)

        if self.use_diffkd:
            self.log("v/diffkd", diffkd_loss, on_step=True, on_epoch=True)

        self.log("train_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        signal, sig_len, transcript, transcript_len, sample_id = batch
        log_probs, enc_len, predictions, _ = self.forward(input_signal=signal, input_signal_length=sig_len)
        transcribed = self.wer.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=log_probs, decoder_lengths=enc_len, return_hypotheses=False
        )
        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().numpy()
        return list(zip(sample_id, transcribed))

def rectified_flow_schedule(t):
    alpha_t = t
    sigma_t = 1 - t
    return alpha_t, sigma_t
def vp_ode_schedule(t, a=19.9, b=0.1):
    alpha_t = torch.exp(-0.25 * a * (1 - t) ** 2 - 0.5 * b * (1 - t))
    sigma_t = torch.sqrt(1 - alpha_t ** 2)
    return alpha_t, sigma_t
def ve_ode_schedule(t, a=0.02, b=100):
    alpha_t = a * (b / a) ** t
    sigma_t = torch.ones_like(t)
    return alpha_t, sigma_t
def rectified_flow_schedule_deriv(t):
    # alpha_t = t, sigma_t = 1 - t
    dalpha_dt = torch.ones_like(t)
    dsigma_dt = -torch.ones_like(t)
    return dalpha_dt, dsigma_dt
def vp_ode_schedule_deriv(t, a=19.9, b=0.1):
    # alpha_t = exp(-0.25 * a * (1-t)^2 - 0.5 * b * (1-t))
    # d(alpha_t)/dt = alpha_t * [0.5*a*(1-t) + 0.5*b]
    alpha_t = torch.exp(-0.25 * a * (1 - t) ** 2 - 0.5 * b * (1 - t))
    dalpha_dt = alpha_t * (0.5 * a * (1 - t) + 0.5 * b)
    # sigma_t = sqrt(1 - alpha_t^2)
    sigma_t = torch.sqrt(1 - alpha_t ** 2)
    dsigma_dt = -alpha_t * dalpha_dt / sigma_t
    return dalpha_dt, dsigma_dt
def ve_ode_schedule_deriv(t, a=0.02, b=100):
    # alpha_t = a * (b/a)^t
    # d(alpha_t)/dt = alpha_t * log(b/a)
    alpha_t = a * (b / a) ** t
    dalpha_dt = alpha_t * torch.log(torch.tensor(b / a, device=t.device, dtype=t.dtype))
    # sigma_t = 1, dsigma_dt = 0
    dsigma_dt = torch.zeros_like(t)
    return dalpha_dt, dsigma_dt

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        layers = []
        if num_layers == 2:
            layers = [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            ]
        elif num_layers == 1:
            layers = [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim),
            ]
        self.encoder = nn.Sequential(*layers)
    def forward(self, x):
        return self.encoder(x)
class SwinTransformerEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4):
        super().__init__()
        self.attn   = nn.MultiheadAttention(embed_dim=in_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.relu    = nn.ReLU()
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        # x: (B, C, T)

        x_seq = x.permute(2, 0, 1)               # [T, B, C]
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)

        attn_out = attn_out.permute(1, 2, 0)     # [B, C, T]

        #    (B, C, T) → (B, T, C)
        h = attn_out.permute(0, 2, 1)            # [B, T, C]
        h = self.linear1(h)                     # [B, T, out_dim]
        h = self.relu(h)
        h = self.linear2(h)                     # [B, T, out_dim]

        return h.permute(0, 2, 1)               # [B, out_dim, T]
class CNNEncoder(nn.Module):
    # for image classification
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 1)
        )
    def forward(self, x):
        return self.block(x)
class UNet1D(nn.Module):
    def __init__(self, in_ch, base_ch, out_ch, num_layers=4):
        super().__init__()
        self.down_channels = []
        ch = in_ch
        self.downs = nn.ModuleList()
        for i in range(num_layers):
            outc = base_ch * (2**i)
            self.downs.append(nn.Conv1d(ch, outc, 4, 2, 1))
            self.down_channels.append(outc)
            ch = outc
        self.bottleneck = nn.Conv1d(ch, ch, 3, 1, 1)
        # up path
        self.ups = nn.ModuleList()
        # reversed skip channels
        for skip_c in reversed(self.down_channels):
            in_c = ch + skip_c
            out_c = skip_c
            self.ups.append(nn.ConvTranspose1d(in_c, out_c, 4, 2, 1))
            ch = out_c
        self.final = nn.Conv1d(ch, out_ch, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        x = self.bottleneck(x)
        for up in self.ups:
            skip = skips.pop()

            if x.size(2) != skip.size(2):
                diff = skip.size(2) - x.size(2)
                x = F.pad(x, (0, diff))
            x = torch.cat([x, skip], dim=1)
            x = up(x)
        return self.final(x)
# ---------------- Conformer Modules ----------------
class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class ConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.pointwise_conv1 = nn.Conv1d(dim, dim * expansion_factor, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            dim * expansion_factor,
            dim * expansion_factor,
            kernel_size=kernel_size,
            groups=dim * expansion_factor,
            padding=kernel_size // 2
        )
        self.batch_norm = nn.BatchNorm1d(dim * expansion_factor)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(dim * expansion_factor, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T) for conv
        x = self.layer_norm(x)
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return self.dropout(x)
class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_mult=4, conv_expansion=2, conv_kernel=31, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, mult=ff_mult, dropout=dropout)
        self.norm_ff1 = nn.LayerNorm(dim)
        self.mha_layer = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.conv_module = ConvModule(dim, expansion_factor=conv_expansion, kernel_size=conv_kernel, dropout=dropout)
        self.ff2 = FeedForwardModule(dim, mult=ff_mult, dropout=dropout)
        self.norm_ff2 = nn.LayerNorm(dim)
        self.norm_final = nn.LayerNorm(dim)

    def forward(self, x):
        # Feed Forward module 1 (half-step)
        residual = x
        x = self.norm_ff1(x)
        x = self.ff1(x)
        x = residual + 0.5 * x

        # Multi-Head Self-Attention
        residual = x
        x = self.mha_layer(x)
        x,_ = self.mha(x, x, x)
        x = residual + x

        # Convolution Module
        residual = x
        x = self.conv_module(x)
        x = residual + x

        # Feed Forward module 2 (half-step)
        residual = x
        x = self.norm_ff2(x)
        x = self.ff2(x)
        x = residual + 0.5 * x

        # Final layer norm
        return self.norm_final(x)
class ConformerEncoder(nn.Module):
    def __init__(self, input_dim, encoder_dim, num_heads, ff_mult=4, conv_expansion_factor=2, num_layers=4, dropout=0.1):
        super().__init__()
        # initial projection if needed
        self.input_proj = nn.Linear(input_dim, encoder_dim) if input_dim != encoder_dim else nn.Identity()
        self.layers = nn.ModuleList([
            ConformerBlock(
                dim=encoder_dim,
                heads=num_heads,
                ff_mult=ff_mult,
                conv_expansion=conv_expansion_factor,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: (B, T, input_dim)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        # return (B, T, encoder_dim)
        return x

class DynamicStepRouter(nn.Module):
    'Select per-sample flow step counts with optional layer embeddings, budget and entropy regularization. Returns steps, router loss, and auxiliary statistics.'
    def __init__(
        self,
        max_steps: int = 16,
        min_steps: int = 1,
        stu_dim: int = None,
        tch_dim: int = None,
        hidden_dim: int = 128,
        proj_dim: int = 128,
        use_layer_id: bool = False,
        num_layers: int = None,
        layer_emb_dim: int = 32,
        feature_reduce: str = "gap",  # 'gap'|'mean'|'last'
        temperature: float = 1.0,
        budget_target: float | None = None,
        budget_weight: float = 0.1,
        entropy_weight: float = 0.0,
        allow_channel_last: bool = True,
    ):
        super().__init__()
        assert 1 <= min_steps <= max_steps
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.K = max_steps
        self.temperature = temperature
        self.use_layer_id = use_layer_id
        self.feature_reduce = feature_reduce
        self.allow_channel_last = allow_channel_last

        assert stu_dim is not None and tch_dim is not None, 'Specify stu_dim and tch_dim.'

        self.stu_proj = nn.Sequential(
            nn.Linear(stu_dim, proj_dim),
            nn.ReLU(inplace=True)
        )
        self.tch_proj = nn.Sequential(
            nn.Linear(tch_dim, proj_dim),
            nn.ReLU(inplace=True)
        )

        if use_layer_id:
            assert num_layers is not None and num_layers > 0
            self.layer_emb = nn.Embedding(num_layers, layer_emb_dim)
            router_in = proj_dim * 2 + layer_emb_dim
        else:
            self.layer_emb = None
            router_in = proj_dim * 2

        self.router = nn.Sequential(
            nn.Linear(router_in, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.K)
        )

        self.budget_target = budget_target
        self.budget_weight = budget_weight
        self.entropy_weight = entropy_weight

        mask = torch.full((self.K,), 0.0)
        if self.min_steps > 1:
            mask[: self.min_steps - 1] = float("-inf")
        self.register_buffer("logit_mask", mask)

    @staticmethod
    def _to_channel_first(x):

        if x.dim() != 3:
            raise ValueError("Feature must be 3D (B, C, T) or (B, T, C).")
        B, A, B_or_T = x.shape

        return x.transpose(1, 2) if A > B_or_T else x

    def _reduce_time(self, x):  # x: (B, C, T)
        if self.feature_reduce == "gap":
            return x.mean(dim=-1)  # (B, C)
        elif self.feature_reduce == "mean":
            return x.mean(dim=-1)
        elif self.feature_reduce == "last":
            return x[..., -1]      # (B, C)
        else:
            raise ValueError(f"unknown feature_reduce: {self.feature_reduce}")

    def forward(self, stu_feat, tch_feat, layer_id=None, temperature: float | None = None, train_mode: bool | None = None):
        if train_mode is None:
            train_mode = self.training
        if temperature is None:
            temperature = self.temperature

        if self.allow_channel_last:

            if stu_feat.shape[1] != stu_feat.shape[2]:
                stu_feat = stu_feat.transpose(1, 2)
            if tch_feat.shape[1] != tch_feat.shape[2]:
                tch_feat = tch_feat.transpose(1, 2)
        else:
            stu_feat = self._to_channel_first(stu_feat)
            tch_feat = self._to_channel_first(tch_feat)

        stu_vec = self._reduce_time(stu_feat)  # (B, Hs)
        tch_vec = self._reduce_time(tch_feat)  # (B, Ht)

        stu_h = self.stu_proj(stu_vec)  # (B, P)
        tch_h = self.tch_proj(tch_vec)  # (B, P)

        if self.use_layer_id:
            if layer_id is None:
                raise ValueError('layer_id is required when use_layer_id=True.')
            if isinstance(layer_id, int):
                layer_id = torch.full((stu_h.size(0),), layer_id, dtype=torch.long, device=stu_h.device)
            lyr = self.layer_emb(layer_id)  # (B, E)
            h = torch.cat([stu_h, tch_h, lyr], dim=-1)
        else:
            h = torch.cat([stu_h, tch_h], dim=-1)

        logits = self.router(h)  # (B, K)

        if self.min_steps > 1:
            logits = logits + self.logit_mask

        probs = F.softmax(logits, dim=-1)  # (B, K)
        expected_steps = (probs * torch.arange(1, self.K + 1, device=probs.device)).sum(dim=-1)  # (B,)

        if train_mode:
            # Gumbel-Softmax (straight-through)
            y_soft = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
            # hard one-hot: straight-through trick
            index = y_soft.argmax(dim=-1)
            y_hard = torch.zeros_like(y_soft).scatter_(1, index.unsqueeze(1), 1.0)
            y = (y_hard - y_soft).detach() + y_soft

            steps = index + 1

            losses = []

            if self.budget_target is not None and self.budget_weight > 0:
                batch_mean = steps.float().mean()
                budget_loss = (batch_mean - self.budget_target) ** 2
                losses.append(self.budget_weight * budget_loss)

            if self.entropy_weight > 0:

                entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()

                losses.append(-self.entropy_weight * entropy)

            router_loss = sum(losses) if len(losses) else logits.new_zeros([])
        else:

            index = probs.argmax(dim=-1)
            steps = index + 1
            router_loss = logits.new_zeros([])

        aux = {
            "logits": logits,
            "probs": probs,
            "expected_steps": expected_steps,
        }
        return steps, router_loss, aux

class FlowMatchingModule(nn.Module):
    def __init__(self, flow_cfg, router: nn.Module = None, router_weight: float = 0.1):
        super().__init__()

        self.meta_encoder_type = flow_cfg.get("meta_encoder_type", "mlp")
        time_embed_dim = flow_cfg.get("time_embed_dim", 32)
        self.hidden_dim = flow_cfg.get("hidden_dim", 96)
        self.training_sampling = flow_cfg.get("training_sampling", 8)
        self.inference_sampling = flow_cfg.get("inference_sampling", 8)
        self.weight = flow_cfg.get("weight", 1.0)
        self.feature_dim = flow_cfg.get("hidden_dim", 96)
        self.teacher_dim = flow_cfg.get("teacher_dim", 176)
        self.student_head_num = flow_cfg.get("student_head_num", 4)
        self.teacher_head_num = flow_cfg.get("teacher_head_num", 8)

        self.router = router  # DepthRouter or None
        self.router_weight = router_weight

        # time embedding
        self.time_embed = nn.Linear(1, time_embed_dim)

        self.meta_encoder = None
        if self.meta_encoder_type == "mlp":
            self.meta_encoder = nn.Sequential(
                nn.Linear(self.feature_dim + time_embed_dim, self.hidden_dim),
                # nn.Linear(self.feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.feature_dim)
            )
        elif self.meta_encoder_type == "cnn":

            self.meta_encoder = nn.Sequential(
                nn.Conv1d(self.feature_dim+time_embed_dim, self.feature_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
            )
        elif self.meta_encoder_type == "swin":
            self.meta_encoder = SwinTransformerEncoder(self.feature_dim + time_embed_dim, self.feature_dim, self.student_head_num)
        elif self.meta_encoder_type == "conformer":

            self.meta_encoder = ConformerEncoder(
                input_dim=self.feature_dim + time_embed_dim,
                encoder_dim=self.feature_dim,
                num_heads=self.student_head_num,
                ff_mult=4,
                conv_expansion_factor=2,
                num_layers=4
            )
        elif self.meta_encoder_type == "unet":

            self.meta_encoder = UNet1D(
                in_ch=self.feature_dim + time_embed_dim,
                base_ch=self.hidden_dim,
                out_ch=self.feature_dim,
                num_layers=4
            )
        else:
            raise ValueError(f"Unknown meta_encoder type: {self.meta_encoder_type}")

        self.shape_transformation_function = None
        shape_transform = flow_cfg.get("shape_transform", "linear")
        self.shape_transform_type = shape_transform
        if shape_transform == "identity":
            self.shape_transformation_function = nn.Identity()
        elif shape_transform == "linear":
            self.shape_transformation_function = nn.Linear(self.feature_dim, self.hidden_dim)
        elif shape_transform == "conv1d":
            self.shape_transformation_function = nn.Conv1d(self.feature_dim, self.hidden_dim, 1)
        else:
            raise ValueError(f"Unknown shape_transform type: {shape_transform}")

        loss_type = flow_cfg.get("loss", "mse") # default: mse
        if loss_type == "mse":
            self.metric_based_loss_function = nn.MSELoss()
        elif loss_type == "cosine":
            self.metric_based_loss_function = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # noise schedule
        self.noise_schedule = None
        noise_schedule = flow_cfg.get("noise_schedule", "rectified")
        if noise_schedule == "rectified":
            self.noise_schedule = rectified_flow_schedule
            self.noise_schedule_deriv = rectified_flow_schedule_deriv
        elif noise_schedule == "vp_ode":
            self.noise_schedule = vp_ode_schedule
            self.noise_schedule_deriv = vp_ode_schedule_deriv
        elif noise_schedule == "ve_ode":
            self.noise_schedule = ve_ode_schedule
            self.noise_schedule_deriv = ve_ode_schedule_deriv
        else:
            raise NotImplementedError

    def forward(self, s_f, t_f=None, target=None, layer_sampling_step: int = None, layer_id: int = None):
        x = s_f
        for i in range(layer_sampling_step, 0, -1):
            # s_f.shape: (32, 179, 88)
            t = torch.full((s_f.size(0), s_f.size(1), 1), i / layer_sampling_step, device=s_f.device)
            # t.shape: (32, 179, 1)
            embed_t = self.time_embed(t)
            # embed_t.shape: (32, 179, 32)
            embed_t = embed_t.permute(0, 2, 1)
            # embed_t.shape: (32, 32, 179)
            if self.meta_encoder_type == "mlp" or self.meta_encoder_type == "conformer":
                x_perm = x.permute(0, 2, 1)
                # x_perm.shape: (batch, 88, 179)
                # embed_t.shape: (batch, 32, 179)
                embed_x = torch.cat([x_perm, embed_t], dim=1)
                # embed_x.shape: (batch, 120, 179)
                embed_x = embed_x.permute(0, 2, 1)
                # embed_x.shape: (32, 179, 120)
                velocity = self.meta_encoder(embed_x)
                # velocity.shape: (32, 179, 88)
            else:
                x_perm = x.permute(0, 2, 1)
                embed_x = torch.cat([x_perm, embed_t], dim=1)
                # embed_x.shape: (batch, 120, 179)
                velocity = self.meta_encoder(embed_x)
                # velocity.shape: (32, 88, 179)
                velocity = velocity.permute(0, 2, 1)
                # velocity.shape: (32, 179, 88)

            if self.meta_encoder_type == "unet":

                T_x = x.size(2)
                T_v = velocity.size(2)
                if T_v != T_x:
                    if T_v > T_x:

                        velocity = velocity[:, :, :T_x]
                    else:

                        pad_amt = T_x - T_v
                        velocity = F.pad(velocity, (0, pad_amt))
            # x: (batch, seq_len, Hs), velocity: (batch, seq_len, Hs)
            x = x - velocity / layer_sampling_step

        loss = 0.0
        if self.training and t_f is not None:
            # t = t.permute(0, 2, 1)
            dalpha_dt, dsigma_dt = self.noise_schedule_deriv(t)
            noise_scheduled_x = (dalpha_dt * s_f - velocity) / (-dsigma_dt)
            if self.shape_transform_type == "linear":
                transformed_s_f = self.shape_transformation_function(noise_scheduled_x)
            else:
                transformed_s_f = self.shape_transformation_function(noise_scheduled_x)
            kd_loss = self.metric_based_loss_function(transformed_s_f, t_f)
            loss = kd_loss

        return loss, x
