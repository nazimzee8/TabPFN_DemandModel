"""Classification-native teacher and loss utilities for MODEL4."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def permute_class_labels(
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    teacher_dict: dict | None,
    rng: "np.random.Generator",
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
    """Apply a random class-label permutation consistently across all tensors.

    The same permutation is applied to y_train, y_test, and (if present) the
    teacher probability/logit columns and coefficient rows/columns so that the
    model cannot use label identity as a shortcut.

    Parameters
    ----------
    y_train, y_test : class index tensors (long).
    teacher_dict : dict with optional keys
        ``teacher_probs`` (N, K), ``teacher_logits`` (N, K),
        ``W_teacher_norm`` (p, K), ``b_teacher`` (K,).
        Pass None or an empty dict to skip teacher remapping.
    rng : numpy default_rng instance.
    num_classes : int K.

    Returns
    -------
    (y_train_perm, y_test_perm, teacher_dict_perm)
        teacher_dict_perm is None when teacher_dict is None.
    """
    perm = rng.permutation(num_classes)
    perm_tensor = torch.as_tensor(perm, dtype=torch.long, device=y_train.device)

    y_train_perm = perm_tensor[y_train.long()]
    y_test_perm  = perm_tensor[y_test.long()]

    if teacher_dict is None:
        return y_train_perm, y_test_perm, None

    td = dict(teacher_dict)

    if td.get("teacher_probs") is not None:
        tp = td["teacher_probs"]
        td["teacher_probs"] = tp[:, perm_tensor.to(tp.device)]

    if td.get("teacher_logits") is not None:
        tl = td["teacher_logits"]
        td["teacher_logits"] = tl[:, perm_tensor.to(tl.device)]

    if td.get("W_teacher_norm") is not None:
        W = td["W_teacher_norm"]
        # W shape (p, K)
        if W.ndim == 2 and W.shape[1] == num_classes:
            td["W_teacher_norm"] = W[:, perm_tensor.to(W.device)]
        elif W.ndim == 2 and W.shape[0] == num_classes:
            td["W_teacher_norm"] = W[perm_tensor.to(W.device), :]

    if td.get("b_teacher") is not None:
        b = td["b_teacher"]
        if b.numel() == num_classes:
            td["b_teacher"] = b[perm_tensor.to(b.device)]

    # Permute categorical class effects if present
    if td.get("alpha_cat") is not None:
        ac = td["alpha_cat"]
        if isinstance(ac, torch.Tensor) and ac.ndim == 3 and ac.shape[2] == num_classes:
            td["alpha_cat"] = ac[:, :, perm_tensor.to(ac.device)]

    return y_train_perm, y_test_perm, td


def inverse_frequency_class_weight(
    y_train: torch.Tensor,
    num_classes: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    counts = torch.bincount(y_train.long(), minlength=num_classes).to(torch.float32)
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].sum() / counts[present].clamp(min=eps)
    if present.any():
        weights[present] = weights[present] / weights[present].mean().clamp(min=eps)
    return weights.to(device=y_train.device)


def canonicalize_class_coefficients(
    W: torch.Tensor,
    b: torch.Tensor | None,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return reference-class coefficients with shapes (p, K) and (K,)."""
    if W.ndim == 1:
        if num_classes != 2:
            raise ValueError("A one-dimensional coefficient vector is valid only for K=2.")
        W_out = torch.zeros(
            W.shape[0], 2, device=W.device, dtype=W.dtype
        )
        W_out[:, 1] = W
    elif W.ndim == 2 and W.shape[1] == num_classes:
        W_out = W
    elif W.ndim == 2 and W.shape[0] == num_classes:
        W_out = W.transpose(0, 1)
    else:
        raise ValueError(
            f"Classification coefficients must have shape (p,), (p,K), or (K,p); "
            f"got {tuple(W.shape)} for K={num_classes}."
        )

    if b is None:
        b_out = torch.zeros(num_classes, device=W.device, dtype=W.dtype)
    elif b.ndim == 0:
        if num_classes != 2:
            raise ValueError("A scalar intercept is valid only for K=2.")
        b_out = torch.zeros(num_classes, device=b.device, dtype=b.dtype)
        b_out[1] = b
    elif b.numel() == num_classes:
        b_out = b.reshape(num_classes)
    else:
        raise ValueError(
            f"Classification intercept must be scalar for K=2 or shape (K,); "
            f"got {tuple(b.shape)}."
        )

    # Softmax parameters are identifiable only up to a shared class offset.
    W_out = W_out - W_out[:, :1]
    b_out = b_out - b_out[:1]
    return W_out, b_out


@dataclass
class LogisticTeacherResult:
    W_teacher_norm: torch.Tensor
    b_teacher: torch.Tensor
    teacher_logits: torch.Tensor
    teacher_probs: torch.Tensor


class LogisticTeacher:
    """Offline sklearn logistic-regression teacher; never called inside model.forward."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        solver: str = "lbfgs",
    ):
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.solver = str(solver)

    @torch.no_grad()
    def fit_predict(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_query: torch.Tensor,
        num_classes: int,
    ) -> LogisticTeacherResult | None:
        if torch.unique(y_train).numel() < 2:
            return None
        try:
            from sklearn.linear_model import LogisticRegression

            kwargs = {
                "C": self.C,
                "max_iter": self.max_iter,
                "solver": self.solver,
                "random_state": 0,
            }
            estimator = LogisticRegression(**kwargs)
            estimator.fit(
                X_train.detach().float().cpu().numpy(),
                y_train.detach().long().cpu().numpy(),
            )

            classes = np.asarray(estimator.classes_, dtype=np.int64)
            p = X_train.shape[1]
            W_np = np.zeros((p, num_classes), dtype=np.float32)
            b_np = np.zeros(num_classes, dtype=np.float32)
            coef = np.asarray(estimator.coef_, dtype=np.float32)
            intercept = np.asarray(estimator.intercept_, dtype=np.float32)
            if coef.shape[0] == 1 and classes.size == 2:
                W_np[:, classes[1]] = coef[0]
                b_np[classes[1]] = intercept[0]
            else:
                for row_idx, class_id in enumerate(classes):
                    if 0 <= int(class_id) < num_classes:
                        W_np[:, int(class_id)] = coef[row_idx]
                        b_np[int(class_id)] = intercept[row_idx]

            W = torch.as_tensor(W_np, device=X_train.device, dtype=X_train.dtype)
            b = torch.as_tensor(b_np, device=X_train.device, dtype=X_train.dtype)
            W, b = canonicalize_class_coefficients(W, b, num_classes)
            logits = X_query @ W + b
            return LogisticTeacherResult(
                W_teacher_norm=W,
                b_teacher=b,
                teacher_logits=logits,
                teacher_probs=torch.softmax(logits, dim=-1),
            )
        except Exception:
            return None


def compute_classification_losses(
    output: dict,
    y_test: torch.Tensor,
    *,
    y_train: torch.Tensor | None = None,
    teacher: LogisticTeacherResult | dict | None = None,
    cfg=None,
    class_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> dict[str, torch.Tensor]:
    logits = output["logits"]
    y_test = y_test.long()
    num_classes = logits.shape[-1]
    label_smoothing = float(getattr(cfg, "class_label_smoothing", 0.0))
    ce = F.cross_entropy(
        logits,
        y_test,
        weight=class_weight,
        label_smoothing=label_smoothing,
    )

    zero = _zero_loss(logits)
    teacher_kl = zero
    coef_aux = zero
    teacher_logits = None
    teacher_W = None
    if teacher is not None:
        if isinstance(teacher, dict):
            teacher_logits = teacher.get("teacher_logits")
            teacher_W = teacher.get("W_teacher_norm")
        else:
            teacher_logits = teacher.teacher_logits
            teacher_W = teacher.W_teacher_norm
    if teacher_logits is not None:
        T = max(float(temperature), 1e-6)
        teacher_kl = F.kl_div(
            F.log_softmax(logits / T, dim=-1),
            F.softmax(teacher_logits.detach() / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
    if teacher_W is not None and output.get("W_hat_norm") is not None:
        W_hat = output["W_hat_norm"]
        W_target = teacher_W.detach().to(device=W_hat.device, dtype=W_hat.dtype)
        W_hat = W_hat - W_hat[:, :1]
        W_target = W_target - W_target[:, :1]
        coef_aux = (W_hat - W_target).pow(2).mean() / (
            W_target.pow(2).mean() + 1e-6
        )

    correct = logits.gather(1, y_test.unsqueeze(1)).squeeze(1)
    wrong_mask = F.one_hot(y_test, num_classes=num_classes).bool()
    strongest_wrong = logits.masked_fill(wrong_mask, float("-inf")).max(dim=1).values
    margin = F.relu(1.0 - correct + strongest_wrong).mean()

    probs = output["probs"]
    if y_train is not None:
        empirical_prior = torch.bincount(
            y_train.long(), minlength=num_classes
        ).to(probs.dtype)
        empirical_prior = empirical_prior / empirical_prior.sum().clamp(min=1.0)
        predicted_prior = probs.mean(dim=0).clamp(min=1e-8)
        prior = F.kl_div(
            predicted_prior.log(),
            empirical_prior.to(predicted_prior.device),
            reduction="sum",
        )
    else:
        prior = zero

    target_one_hot = F.one_hot(y_test, num_classes=num_classes).to(probs.dtype)
    calibration = (probs - target_one_hot).pow(2).sum(dim=1).mean()

    total = (
        float(getattr(cfg, "class_ce_loss_weight", 1.0)) * ce
        + float(getattr(cfg, "class_logit_kl_loss_weight", 0.05)) * teacher_kl
        + float(getattr(cfg, "class_coef_aux_loss_weight", 0.05)) * coef_aux
        + float(getattr(cfg, "class_margin_aux_loss_weight", 0.01)) * margin
        + float(getattr(cfg, "class_prior_aux_loss_weight", 0.01)) * prior
        + float(getattr(cfg, "class_calibration_aux_loss_weight", 0.0)) * calibration
    )
    return {
        "total_loss": total,
        "ce": ce,
        "teacher_kl": teacher_kl,
        "coef_aux": coef_aux,
        "margin": margin,
        "prior": prior,
        "calibration": calibration,
    }
